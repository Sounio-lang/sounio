// C++23 canonical reader/emitter for the frozen original B199 batch0 campaign.
// No p-value/rejection/calibration statistic is calculated by this report.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
using I=std::int64_t;
static void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static std::string read(const fs::path& path){std::ifstream f(path);need(bool(f),"open "+path.string());return {std::istreambuf_iterator<char>(f),{}};}
static std::string trim(std::string s){const auto a=s.find_first_not_of(" \t\r\n"),b=s.find_last_not_of(" \t\r\n");return a==std::string::npos?"":s.substr(a,b-a+1);}
static I integer(const std::string& s){std::size_t used=0;const auto v=std::stoll(trim(s),&used);need(used==trim(s).size(),"not an integer: "+s);return v;}
static I add(I a,I b){need(a>=0&&b>=0&&a<=std::numeric_limits<I>::max()-b,"integer addition overflow");return a+b;}
static I mul(I a,I b){need(a>=0&&b>=0,"negative duration");need(b==0||a<=std::numeric_limits<I>::max()/b,"integer multiplication overflow");return a*b;}
static I centiseconds(const std::string& text){
 std::vector<std::string> parts;std::istringstream in(trim(text));std::string item;
 while(std::getline(in,item,':'))parts.push_back(item);
 need(parts.size()==2||parts.size()==3,"GNU elapsed format");
 const auto tail=parts.back();const auto dot=tail.find('.');
 const auto whole=integer(dot==std::string::npos?tail:tail.substr(0,dot));
 need(whole>=0&&whole<60,"elapsed seconds range");
 I fraction=0;
 if(dot!=std::string::npos){const auto digits=tail.substr(dot+1);need(digits.size()==1||digits.size()==2,"GNU elapsed precision");fraction=integer(digits);if(digits.size()==1)fraction=mul(fraction,10);}
 I seconds=whole;
 if(parts.size()==2)seconds=add(seconds,mul(integer(parts[0]),60));
 else {const auto minute=integer(parts[1]);need(minute>=0&&minute<60,"elapsed minutes range");seconds=add(seconds,add(mul(integer(parts[0]),3600),mul(minute,60)));}
 const auto result=add(mul(seconds,100),fraction);
 need(result<=9007199254740992LL,"duration exceeds exact binary64 integer input");return result;
}
static I wall(const fs::path& path){
 std::istringstream in(read(path));std::string line;bool found=false,exit=false;I value=0;
 while(std::getline(in,line)){
  if(line.find("Elapsed (wall clock) time")!=std::string::npos){need(!found,"duplicate elapsed");const auto at=line.rfind(": ");need(at!=std::string::npos,"elapsed delimiter");value=centiseconds(line.substr(at+2));found=true;}
  if(line.find("Exit status:")!=std::string::npos){need(!exit,"duplicate process exit");need(integer(line.substr(line.rfind(':')+1))==0,"nonzero timed process exit in READY batch");exit=true;}
 }
 need(found&&exit,"incomplete GNU resource");return value;
}
static std::map<std::string,std::string> summary(const fs::path& path){
 std::map<std::string,std::string> out;std::istringstream in(read(path));std::string line;
 while(std::getline(in,line)){const auto tab=line.find('\t');need(tab!=std::string::npos,"summary format");need(out.emplace(line.substr(0,tab),line.substr(tab+1)).second,"duplicate summary key");}
 return out;
}
struct Cell{I n=0;std::string alpha,name;bool ready=false;std::array<I,16> status{},cs{};std::array<I,2> count{0,0},median_twice_cs{-1,-1};};
struct Campaign{std::array<Cell,9> cells;I upper_minutes=1;};
static Campaign load(const fs::path& root){
 Campaign campaign;std::istringstream catalog(read(root/"catalog.tsv"));std::string line;
 need(bool(std::getline(catalog,line))&&line=="index\tn\talpha\tcell\tbatch\tready_at_selection","catalog header");
 for(int index=0;index<9;++index){
  need(bool(std::getline(catalog,line)),"missing catalog cell");std::istringstream row(line);I id=-1,n=0,ready=-1;std::string alpha,name,batch,extra;
  need(bool(row>>id>>n>>alpha>>name>>batch>>ready)&&!(row>>extra),"catalog row");
  const I expected_n=index<3?16:(index<6?64:256);const std::array<std::string,3> alphas{"0",".25","1"};
  need(id==index&&n==expected_n&&alpha==alphas[static_cast<std::size_t>(index%3)]&&name=="n"+std::to_string(n)+"_alpha"+alpha&&batch=="batch0"&&(ready==0||ready==1),"fixed nine-cell catalog/order");
  auto& cell=campaign.cells[static_cast<std::size_t>(index)];cell.n=n;cell.alpha=alpha;cell.name=name;cell.ready=ready==1;cell.status.fill(-1);cell.cs.fill(-1);
  if(!cell.ready)continue;
  const auto dir=root/"raw"/name/"batch0";need(fs::is_regular_file(dir/"READY"),"missing copied READY");
  std::istringstream datasets(read(dir/"datasets.tsv"));need(bool(std::getline(datasets,line))&&line=="index\tn\talpha\tB\tgeneration_seed\tbootstrap_seed","dataset header");
  std::map<I,I> histogram;std::array<std::vector<I>,2> durations;
  for(int j=0;j<16;++j){
   need(bool(std::getline(datasets,line)),"missing dataset");std::istringstream data(line);I di=-1,dn=0,b=0,gs=0,bs=0;long double da=-1;
   need(bool(data>>di>>dn>>da>>b>>gs>>bs)&&!(data>>extra),"dataset row format");
   const long double expected_alpha=index%3==0?0:(index%3==1?.25L:1.L);
   need(di==j&&dn==n&&da==expected_alpha&&b==199,"not original B199 dataset cell");
   const auto stem="case-"+std::to_string(j);need(integer(read(dir/(stem+".status")))==0,"nonzero case process exit");
   std::istringstream raw(read(dir/(stem+".stdout")));I found=0,status=-1,completed=-1;bool complete=false;
   while(std::getline(raw,line)){
    if(line.rfind("CAL_RESULT ",0)==0){
     std::istringstream result(line);std::string tag;I ri=-1,hits=-1,failure=-2,inner=-1,hi=0,lo=0;
     need(bool(result>>tag>>ri>>status>>hits>>completed>>failure>>inner>>hi>>lo)&&!(result>>extra),"CAL_RESULT payload");
     need(ri==j&&status>=0&&completed>=0&&completed<=199&&hits>=0&&hits<=completed,"CAL_RESULT counters");
     if(status==0)need(completed==199&&failure==-1&&inner==0,"available result without completed original target");
     ++found;
    }
    if(line=="COUNT PROFILE CALIBRATION CASE COMPLETE")complete=true;
   }
   need(found==1&&complete,"missing/duplicate terminal case record");
   const I cs=wall(dir/(stem+".resource"));cell.status[static_cast<std::size_t>(j)]=status;cell.cs[static_cast<std::size_t>(j)]=cs;
   const std::size_t group=status==0?0:1;++cell.count[group];durations[group].push_back(cs);++histogram[status];
  }
  while(std::getline(datasets,line))need(trim(line).empty(),"extra dataset outside batch0");
  for(const auto& path:{dir/"summary.tsv",root/"raw"/name/"old-cell-summary.tsv"}){
   const auto old=summary(path);need(integer(old.at("datasets"))==16&&integer(old.at("available"))==cell.count[0]&&integer(old.at("unavailable"))==cell.count[1],"old summary availability mismatch");
   for(const auto& [status,count]:histogram)need(integer(old.at("status_"+std::to_string(status)))==count,"old status histogram mismatch");
  }
  for(std::size_t group=0;group<2;++group){
   auto& v=durations[group];if(v.empty())continue;std::sort(v.begin(),v.end());
   const auto nvalues=v.size();const I twice=nvalues%2?mul(v[nvalues/2],2):add(v[nvalues/2-1],v[nvalues/2]);cell.median_twice_cs[group]=twice;
   campaign.upper_minutes=std::max(campaign.upper_minutes,add(twice/12000,1));
  }
 }
 while(std::getline(catalog,line)){need(trim(line).empty(),"extra catalog cell");}
 return campaign;
}

static void emit(const Campaign& c,const fs::path& output);
static void check(const Campaign& c,const std::string& raw);
int main(int argc,char** argv){
 try{
  if(argc==2&&std::string(argv[1])=="--parser-selftest"){
   need(centiseconds("1:08.47")==((60+8)*100+47),"m:ss parser control");
   need(centiseconds("1:02:03.04")==((3600+2*60+3)*100+4),"h:mm:ss parser control");
   need(centiseconds("0:00.00")==0&&centiseconds("2:03.4")==((2*60+3)*100+40),"zero/fraction parser control");
   for(const std::string s:{"0:60.00","1:60:00.00","-1:00.00","0:00.001"}){bool rejected=false;try{(void)centiseconds(s);}catch(const std::exception&){rejected=true;}need(rejected,"bad elapsed accepted");}
   std::cout<<"CAMPAIGN_RESOURCE_PARSER_SELFTEST_PASS\n";return 0;
  }
  need(argc==4,"usage: REPORT --emit|--check|--negative-selftest FROZEN_RAW_ROOT OUTPUT_SIO|ACTUAL_STDOUT");
  const auto c=load(argv[2]);const std::string mode=argv[1];
  if(mode=="--emit")emit(c,argv[3]);
  else if(mode=="--check"||mode=="--negative-selftest"){
   const auto raw=read(argv[3]);check(c,raw);
   if(mode=="--negative-selftest"){
    std::array<std::string,3> bad{raw,raw,raw};
    const std::regex record(R"re(<!-- CAMPAIGN_ROW ([0-8]) 1 ([^ ]+) ([^ ]+) 1 ([^ ]+) ([01]) ([^ ]+) -->)re");std::smatch row;
    need(std::regex_search(raw,row,record),"negative control needs a nonempty available class");
    bad[0].replace(static_cast<std::size_t>(row.position(2)),static_cast<std::size_t>(row.length(2)),"999");
    bad[1].replace(static_cast<std::size_t>(row.position(4)),static_cast<std::size_t>(row.length(4)),"nan");
    const auto at=bad[2].find("<rect data-source-index="),end=bad[2].find("/>",at);need(at!=std::string::npos&&end!=std::string::npos,"negative control mark");bad[2].erase(at,end+2-at);
    for(std::size_t i=0;i<bad.size();++i){bool rejected=false;try{check(c,bad[i]);}catch(const std::exception&){rejected=true;}need(rejected,"undetected campaign report mutant");std::cout<<"CAMPAIGN_REPORT_MUTANT_REJECTED kind="<<i<<'\n';}
   }
   std::cout<<"COUNT_PROFILE_CAMPAIGN_REPORT_PASS cells=9 original_B=199 batch=0 no_pooling=1\n";
  }
  else {throw std::runtime_error("unknown mode");}
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

static void emit(const Campaign& c,const fs::path& output){
 std::ofstream sio(output);need(bool(sio),"open generated Sounio");
 sio<<R"SIO(// Generated by count_profile_campaign_report_reference.cpp --emit.
// Frozen original qualification_B199, first READY batch0 only; no future refresh.
// Raw unavailable-cell slots are -1 and ignored, never interpreted as observations.
//@ requires: madaros
//@ expect-stdout-contains: CAMPAIGN_REPORT_COMPLETE
use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_set,numeric_buffer_at,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_set,integer_buffer_at,integer_buffer_close}
use stats::dynamic_summaries::{dynamic_summary_order,dynamic_quantile_sorted}
use viz::scientific_document::{ScientificDocumentText,ScientificOverlayPlan,ScientificOverlayText,scientific_two_panel_plan,scientific_overlay_validate_sized,scientific_overlay_canvas_sized,scientific_overlay_svg_sized}
use display::canvas::{Canvas}
use viz::scatter::{scatter_project}
use io::sci_print::{print_f64_sci}
// Optional exact native-raster export; Linux native only. No environment logging.
fn scientific_canvas_dump_enabled()->bool with IO,Mut,Div,Panic {
 let data=read_file("/proc/self/environ")
 let wanted="SOUNIO_VIZ_CANVAS_DUMP=1"
 let wanted_len=str_len(wanted)
 var size:i64=0;var found_end=false
 while size<1048575 {
  if data[size as usize]==0 {
   if data[(size+1) as usize]==0 {found_end=true;break}
  }
  size=size+1
 }
 if !found_end {return false}
 var at:i64=0
 while at<size {
  var end=at
  while end<size {if data[end as usize]==0 {break};end=end+1}
  if end-at==wanted_len {
   var same=true;var j:i64=0
   while j<wanted_len {
    if data[(at+j) as usize]!=(str_char_at(wanted,j) as i8) {same=false}
    j=j+1
   }
   if same {return true}
  }
  at=end+1
 }
 false
}
fn scientific_canvas_run(start:i64,length:i64,color:i64) with IO {
 print("CANVAS_RUN ");print(start);print(" ");print(length);print(" ");println(color)
}
fn scientific_canvas_dump(c:&Canvas) with IO,Mut,Div,Panic {
 if !scientific_canvas_dump_enabled() {return}
 let width:i64=c.width;let height:i64=c.height
 assert(width>0 && width<=4096 && height>0 && height<=4096)
 let total:i64=width*height
 assert(total<=16777216 && (c.pixels as i64)>0)
 print("CANVAS_SIZE ");print(width);print(" ");println(height)
 var start:i64=0;var color:i64=read_i64(c.pixels,0);var i:i64=1
 while i<total {
  let next:i64=read_i64(c.pixels,i)
  if next!=color {scientific_canvas_run(start,i-start,color);start=i;color=next}
  i=i+1
 }
 scientific_canvas_run(start,total-start,color)
 println("CANVAS_END")
}

fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 sio<<" let raw_ready:[i64;9]=[";
 for(std::size_t i=0;i<9;++i){if(i)sio<<',';sio<<(c.cells[i].ready?1:0);}sio<<"]\n let raw_status:[i64;144]=[";
 for(std::size_t i=0;i<144;++i){if(i)sio<<',';sio<<c.cells[i/16].status[i%16];}sio<<"]\n let raw_cs:[i64;144]=[";
 for(std::size_t i=0;i<144;++i){if(i)sio<<',';sio<<c.cells[i/16].cs[i%16];}sio<<"]\n let upper_minutes:i64="<<c.upper_minutes<<'\n';
 sio<<R"SIO(
 var x=numeric_buffer_new(9,72);var count_a=numeric_buffer_new(9,72);var count_b=numeric_buffer_new(9,72)
 var median_a=numeric_buffer_new(9,72);var median_b=numeric_buffer_new(9,72)
 var ready=integer_buffer_new(9,72);var mask_a=integer_buffer_new(9,72);var mask_b=integer_buffer_new(9,72)
 var time=numeric_buffer_new(16,128);var selection=integer_buffer_new(16,128);var work=numeric_buffer_new(16,128)
 var empty=integer_buffer_new(0,0)
 var i:i64=0
 while i<9 {
  if raw_ready[i as usize]!=0 && raw_ready[i as usize]!=1 {return 1}
  let z0=numeric_buffer_set(&!x,i,i as f64)
  let z1=numeric_buffer_set(&!count_a,i,0.0);let z2=numeric_buffer_set(&!count_b,i,0.0)
  let z3=numeric_buffer_set(&!median_a,i,0.0);let z4=numeric_buffer_set(&!median_b,i,0.0)
  let z5=integer_buffer_set(&!ready,i,raw_ready[i as usize]);let z6=integer_buffer_set(&!mask_a,i,0);let z7=integer_buffer_set(&!mask_b,i,0)
  if raw_ready[i as usize]==1 {
   var ca:i64=0;var cb:i64=0;var j:i64=0
   while j<16 {
    let pos=i*16+j;let status=raw_status[pos as usize];let cs=raw_cs[pos as usize]
    if status<0 || cs<0 || cs>9007199254740992 {return 1}
    if status==0 {ca=ca+1}else{cb=cb+1}
    let minutes=(cs as f64)/6000.0
    let put=numeric_buffer_set(&!time,j,minutes);j=j+1
   }
   let qa=numeric_buffer_set(&!count_a,i,ca as f64);let qb=numeric_buffer_set(&!count_b,i,cb as f64)
   var kind:i64=0
   while kind<2 {
    j=0
    while j<16 {
     let status=raw_status[(i*16+j) as usize]
     let active=if kind==0 {if status==0 {1}else{0}}else{if status==0 {0}else{1}}
     let put=integer_buffer_set(&!selection,j,active);j=j+1
    }
    let expected=if kind==0 {ca}else{cb}
    let ordered=dynamic_summary_order(&time,&selection,&!work)
    if expected==0 {if ordered.status!=4 {return 1}}
    else {
     if ordered.status!=0 || ordered.available!=expected {return 1}
     let q=dynamic_quantile_sorted(&work,ordered.available,0.5)
     if q.status!=0 || q.value<0.0 || q.value>(upper_minutes as f64) {return 1}
     if kind==0 {
      let value=numeric_buffer_set(&!median_a,i,q.value);let flag=integer_buffer_set(&!mask_a,i,1)
     } else {
      let value=numeric_buffer_set(&!median_b,i,q.value);let flag=integer_buffer_set(&!mask_b,i,1)
     }
    }
    kind=kind+1
   }
  }
  i=i+1
 }
)SIO";
 const std::array<std::string,8> texts{"Synthetic partial B199 campaign - original first READY batch0 only","cell index (x); datasets (count,y); CAL_RESULT status0 is available","cell index (x); median process wall MINUTES (y); early exits included","Cells0..2:n16;3..5:n64;6..8:n256; alpha0,.25,1. Missing/no-class masked. Partial/unassessed; no calibration or algorithm comparison.","result available","failed / unresolved","result available","failed / unresolved"};
 std::string joined;std::array<std::size_t,8> starts{};
 for(std::size_t i=0;i<texts.size();++i){starts[i]=joined.size();joined+=texts[i];}
 sio<<" let chars:[i8;"<<joined.size()<<"]=[";
 for(std::size_t i=0;i<joined.size();++i){if(i)sio<<',';sio<<static_cast<unsigned int>(static_cast<unsigned char>(joined[i]));}
 sio<<"]\n var bytes=integer_buffer_new("<<joined.size()<<','<<joined.size()<<"*8)\n i=0\n while i<"<<joined.size()<<" {let put=integer_buffer_set(&!bytes,i,chars[i as usize] as i64);i=i+1}\n";
 sio<<" let text=ScientificDocumentText{title_start:"<<starts[0]<<",title_len:"<<texts[0].size()<<",left_start:"<<starts[1]<<",left_len:"<<texts[1].size()<<",right_start:"<<starts[2]<<",right_len:"<<texts[2].size()<<",footer_start:"<<starts[3]<<",footer_len:"<<texts[3].size()<<"}\n";
 sio<<" let labels=ScientificOverlayText{document:text,left_first_start:"<<starts[4]<<",left_first_len:"<<texts[4].size()<<",left_second_start:"<<starts[5]<<",left_second_len:"<<texts[5].size()<<",right_first_start:"<<starts[6]<<",right_first_len:"<<texts[6].size()<<",right_second_start:"<<starts[7]<<",right_second_len:"<<texts[7].size()<<"}\n";
 sio<<R"SIO(
 let plan=scientific_two_panel_plan(1600,640,0.0,8.0,0.0,16.0,0.0,8.0,0.0,upper_minutes as f64)
 if plan.status!=0 {return 1}
 let overlay=ScientificOverlayPlan{document:plan,left_kind:0,right_kind:0}
 let valid=scientific_overlay_validate_sized(&overlay,&labels,&bytes,&x,&count_a,&ready,&x,&median_a,&mask_a,&x,&count_b,&ready,&empty,0,&x,&median_b,&mask_b,&empty,0,3,3)
 if valid!=0 {return 1}
 let pixels=heap_alloc(1600*640*8) as *mut i64
 if (pixels as i64)<=0 {return 1}
 var canvas=Canvas{pixels:pixels,width:1600,height:640}
 let drawn=scientific_overlay_canvas_sized(&!canvas,&overlay,&labels,&bytes,&x,&count_a,&ready,&x,&median_a,&mask_a,&x,&count_b,&ready,&empty,0,&x,&median_b,&mask_b,&empty,0,3,3)
 if drawn.status!=0 {heap_free(canvas.pixels as *mut i8);return 1}
 // Structured XML comments preserve numerical values without invalidating SVG XML.
 print("<!-- CAMPAIGN_UPPER ");print(upper_minutes);println(" -->")
 i=0
 while i<9 {
  let cell_ready:i64=raw_ready[i as usize]
  print("<!-- CAMPAIGN_ROW ");print(i);print(" ");print(cell_ready);print(" ")
  if raw_ready[i as usize]==1 {
   let a:i64=numeric_buffer_at(&count_a,i) as i64;let b:i64=numeric_buffer_at(&count_b,i) as i64
   print(a);print(" ");print(b)
  } else {print("NA NA")}
  let a:i64=integer_buffer_at(&mask_a,i);let b:i64=integer_buffer_at(&mask_b,i)
  print(" ");print(a);print(" ")
  if a==1 {let value:f64=numeric_buffer_at(&median_a,i);print_f64_sci(value,15)}else{print("NA")}
  print(" ");print(b);print(" ")
  if b==1 {let value:f64=numeric_buffer_at(&median_b,i);print_f64_sci(value,15)}else{print("NA")}
  println(" -->");i=i+1
 }
 var probe:i64= -1;i=0
 while i<9 {if probe<0 {if raw_ready[i as usize]==1 {probe=i}};i=i+1}
 if probe>=0 {
  let value:f64=numeric_buffer_at(&count_b,probe)
  let point=scatter_project(probe as f64,value,0.0,8.0,0.0,16.0,plan.left.x,plan.left.y,plan.left.w,plan.left.h,true)
  if point.status!=0 {heap_free(canvas.pixels as *mut i8);return 1}
  let px:i64=point.x;let py:i64=point.y;let color:i64=read_i64(canvas.pixels,py*1600+px)
  print("<!-- CAMPAIGN_PIXEL ");print(px);print(" ");print(py);print(" ");print(color);println(" -->")
 }
 let shown=scientific_overlay_svg_sized(&overlay,&labels,&bytes,&x,&count_a,&ready,&x,&median_a,&mask_a,&x,&count_b,&ready,&empty,0,&x,&median_b,&mask_b,&empty,0,3,3)
 if shown.status!=0 {heap_free(canvas.pixels as *mut i8);return 1}
 println("<!-- CAMPAIGN_REPORT_COMPLETE -->")
 scientific_canvas_dump(&canvas)
 heap_free(canvas.pixels as *mut i8)
 numeric_buffer_close(x);numeric_buffer_close(count_a);numeric_buffer_close(count_b);numeric_buffer_close(median_a);numeric_buffer_close(median_b)
 integer_buffer_close(ready);integer_buffer_close(mask_a);integer_buffer_close(mask_b);integer_buffer_close(empty)
 numeric_buffer_close(time);integer_buffer_close(selection);numeric_buffer_close(work);integer_buffer_close(bytes)
 0
}
)SIO";
 sio.close();need(bool(sio),"write generated Sounio");
 std::cout<<"cell\tn\talpha\tready\tavailable\tunavailable\tmedian_available_minutes\tmedian_unavailable_minutes\n";
 for(std::size_t i=0;i<9;++i){
  const auto& x=c.cells[i];std::cout<<i<<'\t'<<x.n<<'\t'<<x.alpha<<'\t'<<x.ready<<'\t';
  if(x.ready)std::cout<<x.count[0]<<'\t'<<x.count[1];else std::cout<<"NA\tNA";
  for(int group=0;group<2;++group){std::cout<<'\t';if(x.median_twice_cs[group]<0)std::cout<<"NA";else std::cout<<std::setprecision(21)<<static_cast<long double>(x.median_twice_cs[group])/12000.L;}
  std::cout<<'\n';
 }
 std::cout<<"upper_minutes\t"<<c.upper_minutes<<"\n";
}
struct Range{I lo,hi;};
static Range project(long double v,long double error,long double domain,I origin,I extent,bool vertical){
 const auto a=static_cast<I>(std::trunc(std::clamp((v-error)/domain*extent,0.L,static_cast<long double>(extent))));
 const auto b=static_cast<I>(std::trunc(std::clamp((v+error)/domain*extent,0.L,static_cast<long double>(extent))));
 const Range out=vertical?Range{origin+extent-b,origin+extent-a}:Range{origin+a,origin+b};need(out.hi-out.lo<=1,"forward pixel budget too wide");return out;
}
static long double numeric_budget(long double v){return 1e-12L*(1+std::fabs(v));}
static I label_width(long double v){
 if(v==0)return 8;
 std::ostringstream out;out<<std::scientific<<std::setprecision(5)<<static_cast<double>(v);
 const auto s=out.str();const auto at=s.find('e');need(at!=std::string::npos,"independent scientific label format");
 const int exponent=std::stoi(s.substr(at+1));const auto digits=std::to_string(std::abs(exponent)).size();return static_cast<I>(9+digits)*8;
}
static std::string group(const std::string& svg,const std::string& id){
 const auto a=svg.find("<g id=\""+id+"\">"),b=svg.find("</g>",a);need(a!=std::string::npos&&b!=std::string::npos,"missing report layer");return svg.substr(a,b-a);
}
static void check(const Campaign& c,const std::string& raw){
 std::array<std::array<long double,2>,9> medians{};std::array<bool,9> seen{};
 const std::regex row(R"re(<!-- CAMPAIGN_ROW ([0-8]) ([01]) ([^ ]+) ([^ ]+) ([01]) ([^ ]+) ([01]) ([^ ]+) -->)re");
 I rows=0;
 for(std::sregex_iterator i(raw.begin(),raw.end(),row),e;i!=e;++i){
  const auto& m=*i;const auto index=integer(m[1]);need(index==rows&&!seen[static_cast<std::size_t>(index)],"report row order/duplicate");seen[static_cast<std::size_t>(index)]=true;
  const auto& cell=c.cells[static_cast<std::size_t>(index)];need(integer(m[2])==(cell.ready?1:0),"missing cell promoted to available");
  for(int cls=0;cls<2;++cls){
   const std::string count=m[3+cls];
   if(cell.ready)need(integer(count)==cell.count[static_cast<std::size_t>(cls)],"raw/old-summary count mismatch in Sounio");else need(count=="NA","missing cell emitted numerical zero count");
   const auto flag=integer(m[cls==0?5:7]);const std::string value=m[cls==0?6:8];
   const bool available=cell.ready&&cell.count[static_cast<std::size_t>(cls)]>0;
   need(flag==(available?1:0),"no-class median mask");
   if(!available){need(value=="NA","missing median represented by number");continue;}
   const long double expected=static_cast<long double>(cell.median_twice_cs[static_cast<std::size_t>(cls)])/12000.L;
   std::size_t used=0;const long double actual=std::stold(value,&used);
   need(used==value.size()&&std::isfinite(actual)&&std::fabs(actual-expected)<=numeric_budget(expected),"independent raw-time median mismatch before projection");
   medians[static_cast<std::size_t>(index)][static_cast<std::size_t>(cls)]=expected;
  }
  ++rows;
 }
 need(rows==9,"incomplete numerical report");
 const std::regex upper(R"(<!-- CAMPAIGN_UPPER ([0-9]+) -->)");std::smatch u;
 need(std::regex_search(raw,u,upper)&&integer(u[1])==c.upper_minutes,"caller viewport upper rule");
 const auto begin=raw.find("<svg "),finish=raw.find("</svg>",begin);
 need(begin!=std::string::npos&&finish!=std::string::npos&&raw.find("<svg ",begin+1)==std::string::npos,"one complete report SVG");
 const auto svg=raw.substr(begin,finish+6-begin);
 need(svg.find("width=\"1600\" height=\"640\"")!=std::string::npos,"report dimensions");
 for(const auto& label:{"Synthetic partial B199 campaign - original first READY batch0 only","CAL_RESULT status0 is available","median process wall MINUTES","early exits included","Cells0..2:n16;3..5:n64;6..8:n256; alpha0,.25,1.","Missing/no-class masked","Partial/unassessed; no calibration or algorithm comparison.","result available","failed / unresolved"})need(svg.find(label)!=std::string::npos,std::string("missing scientific declaration: ")+label);
 need(svg.find("data-interval-kind=\"none\"")!=std::string::npos&&svg.find("data-scientific-overlay-complete=\"true\"")!=std::string::npos,"invented interval or incomplete report");
 need(svg.find("data-primary-radius=\"3\" data-secondary-radius=\"3\"")!=std::string::npos,"report's declared marker radius missing");
 std::set<std::string> ids;const std::regex id(R"re(\bid="([^"]+)")re");
 for(std::sregex_iterator i(svg.begin(),svg.end(),id),e;i!=e;++i){need(ids.insert((*i)[1]).second,"duplicate ID");}
 need(ids.size()==4,"report layer IDs");
 const std::regex text(R"re(<text [^>]*x="(-?[0-9]+)" y="(-?[0-9]+)"[^>]*textLength="([0-9]+)"[^>]*>([^<]*)</text>)re");
 I text_count=0;
 for(std::sregex_iterator i(svg.begin(),svg.end(),text),e;i!=e;++i){
  const auto x=integer((*i)[1]),y=integer((*i)[2]),w=integer((*i)[3]);need(x>=0&&x+w<=1600&&y-7>=0&&y+1<=640,"clipped report label");
  need(w==static_cast<I>((*i)[4].length())*8,"label cell width");++text_count;
 }
 need(text_count==28,"report label/legend count");
 const std::regex legend(R"re(<rect data-overlay-legend="scatter" x="[-0-9.]+" y="[-0-9.]+" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+,[0-9]+,[0-9]+)\)"/>)re");
 const std::array<std::string,4> colors{"70,180,235","235,130,70","70,180,235","235,130,70"};std::size_t legends=0;
 for(std::sregex_iterator i(svg.begin(),svg.end(),legend),e;i!=e;++i){need(legends<4&&(*i)[3]==colors[legends],"availability color identity");need((*i)[1]=="7"&&(*i)[2]=="7","legend/data size mismatch");++legends;}need(legends==4,"missing legend");
 I yw=0,xw=0;
 for(int i=0;i<=4;++i){yw=std::max({yw,label_width(16.L*i/4),label_width(static_cast<long double>(c.upper_minutes)*i/4)});xw=std::max(xw,label_width(8.L*i/4));}
 const I ox=24+yw,rx=800+ox,w=800-ox-(24+xw/2),oy=100,h=400;
 const std::array<std::string,4> names{"scientific-layer-left-primary","scientific-layer-left-secondary","scientific-layer-right-primary","scientific-layer-right-secondary"};
 const std::regex rect(R"re(<rect data-source-index="([0-9]+)" data-source-row="-1" data-source-col="-1" x="([-0-9.]+)" y="([-0-9.]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+,[0-9]+,[0-9]+)\)"/>)re");
 I marks=0;
 for(int layer=0;layer<4;++layer){
  const bool right=layer>=2;const std::size_t cls=static_cast<std::size_t>(layer%2);const auto body=group(svg,names[static_cast<std::size_t>(layer)]);
  auto actual=std::sregex_iterator(body.begin(),body.end(),rect),e=std::sregex_iterator();const I x=right?rx:ox;
  for(std::size_t index=0;index<9;++index){
   const auto& cell=c.cells[index];if(!cell.ready||(right&&cell.count[cls]==0))continue;
   need(actual!=e,"missing report point");const auto& m=*actual;need(integer(m[1])==static_cast<I>(index)&&m[6]==colors[static_cast<std::size_t>(layer)],"source cell provenance/RGB");
   const long double value=right?medians[index][cls]:static_cast<long double>(cell.count[cls]);
   const auto cx=project(static_cast<long double>(index),0,8,x,w,false);
   const auto cy=project(value,right?numeric_budget(value):0,right?static_cast<long double>(c.upper_minutes):16.L,oy,h,true);
   const long double ax=std::stold(m[2])+0.5L,ay=std::stold(m[3])+0.5L;need(ax==std::trunc(ax)&&ay==std::trunc(ay),"report pixelcell origin");bool match=false;
   for(I px=cx.lo;px<=cx.hi;++px)for(I py=cy.lo;py<=cy.hi;++py){
    const I left=std::max(x,px-3),top=std::max(oy,py-3),rw=std::min(x+w+1,px+4)-left,rh=std::min(oy+h+1,py+4)-top;
    if(ax==left&&ay==top&&integer(m[4])==rw&&integer(m[5])==rh)match=true;
   }
   need(match,"report point geometry");++actual;++marks;
  }
  need(actual==e,"extra point: missing cell/class plotted as zero");
 }
 const auto first=std::find_if(c.cells.begin(),c.cells.end(),[](const Cell& x){return x.ready;});
 if(first!=c.cells.end()){
  const std::regex pixel(R"(<!-- CAMPAIGN_PIXEL ([0-9]+) ([0-9]+) ([0-9]+) -->)");std::smatch p;need(std::regex_search(raw,p,pixel),"missing actual Canvas report pixel");
  const auto index=static_cast<I>(std::distance(c.cells.begin(),first));const auto px=project(index,0,8,ox,w,false),py=project(first->count[1],0,16,oy,h,true);
  need(integer(p[1])>=px.lo&&integer(p[1])<=px.hi&&integer(p[2])>=py.lo&&integer(p[2])<=py.hi&&integer(p[3])==((235LL<<16)|(130LL<<8)|70),"actual Canvas count/RGB");
 }
 need(raw.find("<!-- CAMPAIGN_REPORT_COMPLETE -->")!=std::string::npos,"native completion missing");
 std::cout<<"CAMPAIGN_GEOMETRY_VERIFIED marks="<<marks<<" text_bounds="<<text_count<<" numeric_cells=9\n";
}

// C++23 independent named-table framing/concordance and fixture generator.
#include <algorithm>
#include <bit>
#include <cmath>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using I=std::int64_t;
constexpr I prime=1000000007;
I hash_step(I h,I value){return (h*131+value%prime)%prime;}
struct Case{std::string text;I samples=2,max_id=1024,status=0,position=-1,headers=0,rows=0,id_hash=0,count_hash=0;};
void expected_valid(Case& c){
 std::istringstream in(c.text);std::string line;bool header=true;
 while(std::getline(in,line)){
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  std::vector<std::string> fields;std::size_t at=0;
  for(;;){auto end=line.find('\t',at);fields.push_back(line.substr(at,end-at));if(end==std::string::npos)break;at=end+1;}
  if(fields.size()!=static_cast<std::size_t>(c.samples+1))throw std::runtime_error("fixture layout");
  if(header){for(std::size_t i=1;i<fields.size();++i){for(unsigned char b:fields[i])c.id_hash=hash_step(c.id_hash,b);c.id_hash=hash_step(c.id_hash,0);++c.headers;}header=false;}
  else {for(unsigned char b:fields[0])c.id_hash=hash_step(c.id_hash,b);c.id_hash=hash_step(c.id_hash,0);for(std::size_t i=1;i<fields.size();++i){I value=0;auto r=std::from_chars(fields[i].data(),fields[i].data()+fields[i].size(),value);if(r.ec!=std::errc{}||r.ptr!=fields[i].data()+fields[i].size())throw std::runtime_error("fixture count");c.count_hash=hash_step(c.count_hash,value);}++c.rows;}
 }
 c.position=static_cast<I>(c.text.size());
}
std::vector<Case> cases(){
 std::vector<Case> out;
 out.push_back({"gene\tA\tB\r\nx\t0\t9223372036854775807\r\ny\t0004\t12"});
 out.push_back({"\t A\tA \ng\xC3\xA9\t0\t3\ng" "e\xCC\x81\t4\t5\n"});
 out.push_back({"gene\tA\tB\n"+std::string(255,'q')+"\t1\t2\r\n"+std::string(256,'r')+"\t3\t4\n"});
 Case wide;wide.samples=512;wide.text="feature";for(I i=0;i<wide.samples;++i)wide.text+="\ts"+std::to_string(i);wide.text+="\r\n";for(I g=0;g<3;++g){wide.text+="g"+std::to_string(g);for(I s=0;s<wide.samples;++s)wide.text+='\t'+std::to_string((g+1)*(s+1));wide.text+=g==2?"":"\r\n";}out.push_back(wide);
 for(auto& c:out)expected_valid(c);
 auto bad=[&](std::string text,I code,I pos,I maxid=1024){Case c;c.text=std::move(text);c.status=code;c.position=pos;c.max_id=maxid;out.push_back(std::move(c));};
 const std::string base="g\tA\tB\nx\t";
 bad(base+"9223372036854775808\t0\n",5,static_cast<I>(base.size()+18));
 bad(base+"-1\t0\n",3,static_cast<I>(base.size()));
 bad(base+"1\t\n",3,static_cast<I>(base.size()+2));
 bad(base+"1\n",4,static_cast<I>(base.size()+1));
 bad(base+"1\t2\t3\n",4,static_cast<I>(base.size()+3));
 bad("g\tA\tB\r",3,6);
 bad("g\tA\tB\rx",3,6);
 bad("g\t\tB\n",3,2);
 bad("g\tA\tB\n\t1\t2\n",3,6);
 bad("g\tA\tB\nxx\t1\t2\n",8,7,1);
 std::string nul=base+"1\t2\n";nul[6]=0;bad(nul,3,6);
 return out;
}
void generate(){
 std::cout<<R"SIO(use collections::integer_buffer::{integer_buffer_new,integer_buffer_at,integer_buffer_close}
use genomics::io::named_count_tsv::{named_count_tsv_new,named_count_tsv_feed,named_count_tsv_ack,named_count_tsv_finish}
fn gh(h:i64,v:i64)->i64 with Div {(h*131+v%1000000007)%1000000007}
)SIO";
 auto all=cases();
 for(std::size_t k=0;k<all.size();++k){auto& c=all[k];
 for(std::size_t first=0;first<c.text.size();first+=256){const auto n=std::min<std::size_t>(256,c.text.size()-first);std::cout<<"fn named_bytes_"<<k<<'_'<<first/256<<"(i:i64)->i64 with Panic {let raw:[i64;"<<n<<"]=[";for(std::size_t i=0;i<n;++i){if(i)std::cout<<',';std::cout<<static_cast<unsigned>(static_cast<unsigned char>(c.text[first+i]));}std::cout<<"];raw[(i-"<<first<<") as usize]}\n";}
 std::cout<<"fn named_byte_"<<k<<"(i:i64)->i64 with Panic {\n";for(std::size_t first=0;first<c.text.size();first+=256)std::cout<<" if i<"<<std::min(first+256,c.text.size())<<" {return named_bytes_"<<k<<'_'<<first/256<<"(i)}\n";std::cout<<" -1\n}\n";
 std::cout<<"fn parser_case_"<<k<<"(step:i64) with Alloc,Mut,Div,Panic,IO {\n var state=named_count_tsv_new("<<c.samples<<","<<c.max_id<<")\n var id=integer_buffer_new("<<c.max_id<<","<<c.max_id*8<<")\n var row=integer_buffer_new("<<c.samples<<","<<c.samples*8<<")\n";
 std::cout<<R"SIO( var chunk:[i64;256]=[0;256]
 var position:i64=0;var headers:i64=0;var rows:i64=0;var ids:i64=0;var counts:i64=0
 var code:i64=0;var error:i64=0;var done:bool=false
 while !done {
  var n:i64=0
  while n<step && position+n<)SIO"<<c.text.size()<<R"SIO( {chunk[n as usize]=named_byte_)SIO"<<k<<R"SIO((position+n);n=n+1}
  let event=if n>0 {named_count_tsv_feed(&!state,&chunk,n,&!id,&!row)}else{named_count_tsv_finish(&!state,&id,&!row)}
  position=position+event.consumed;code=event.status;error=event.error_position
  if code!=0 {done=true}else{
   if event.kind!=0 {
    var j:i64=0
    while j<event.id_len {ids=gh(ids,integer_buffer_at(&id,j));j=j+1}
    ids=gh(ids,0)
    if event.kind==1 {headers=headers+1}else{rows=rows+1;j=0;while j<)SIO"<<c.samples<<R"SIO( {counts=gh(counts,integer_buffer_at(&row,j));j=j+1}}
    assert(named_count_tsv_ack(&!state))
   }else if event.finished {done=true}
  }
 }
)SIO";
 std::cout<<" println(\"NAMED "<<k<<" {} {} {} {} {} {} {}\",step,code,position,headers,rows,ids,counts)\n if code!=0 {assert(position==error)}\n integer_buffer_close(id);integer_buffer_close(row)\n}\n";
 }
 std::cout<<"fn main() with Alloc,Mut,Div,Panic,IO {\n var step:i64=1\n while step<=256 {\n";
 for(std::size_t k=0;k<all.size();++k)std::cout<<" parser_case_"<<k<<"(step)\n";
 std::cout<<" step=step+1\n }\n println(\"NAMED_ROWS_COMPLETE\")\n}\n";
}
int verify(const char* path){
 std::ifstream in(path);if(!in)return 2;auto all=cases();std::vector<std::vector<bool>> seen(all.size(),std::vector<bool>(257));std::string line;bool complete=false;I rows=0;
 while(std::getline(in,line)){
  if(line=="NAMED_ROWS_COMPLETE"){if(complete)return 9;complete=true;continue;}
  std::istringstream row(line);std::string tag,extra;I k,step,status,position,headers,nrows,ids,counts;
  if(!(row>>tag>>k>>step>>status>>position>>headers>>nrows>>ids>>counts)||tag!="NAMED"||(row>>extra)||k<0||k>=static_cast<I>(all.size())||step<1||step>256||seen[k][step])return 9;
  auto& c=all[k];seen[k][step]=true;++rows;
  if(status!=c.status||position!=c.position||(status==0&&(headers!=c.headers||nrows!=c.rows||ids!=c.id_hash||counts!=c.count_hash))){std::cerr<<"named framing/identity/count disagreement "<<k<<' '<<step<<" expected "<<c.status<<' '<<c.position<<" actual "<<status<<' '<<position<<'\n';return 8;}
 }
 if(!complete||rows!=static_cast<I>(all.size()*256))return 9;
 std::cout<<"NAMED_REFERENCE_PASS cases="<<all.size()<<" chunk_sizes=256 rows="<<rows<<'\n';return 0;
}

int verify_csr(const char* path){
 using B=boost::multiprecision::cpp_dec_float_100;
 std::vector<std::vector<B>> ratios(3);const I counts[3][3]={{5,0,0},{0,7,0},{4,0,6}};
 for(int g=0;g<3;++g){B mean=0;int n=0;for(int col=0;col<3;++col)if(counts[g][col]>0){mean+=log(B(counts[g][col]));++n;}mean/=n;for(int col=0;col<3;++col)if(counts[g][col]>0)ratios[col].push_back(log(B(counts[g][col]))-mean);}
 std::vector<B> logs;B center=0;for(auto& v:ratios){std::sort(v.begin(),v.end());B median=v[v.size()/2];if(v.size()%2==0)median=(v[v.size()/2-1]+median)/2;logs.push_back(median);center+=median;}center/=3;for(auto& v:logs)v-=center;
 std::ifstream in(path);if(!in)return 2;std::string line;bool complete=false;bool status_seen[2]={};bool seen[2][3]={};int n=0;
 while(std::getline(in,line)){
  if(line=="CSR_SMOKE_PASS"){if(complete)return 9;complete=true;continue;}
  std::istringstream row(line);std::string tag,extra;I mode,a,b,c;
  if(!(row>>tag>>mode>>a>>b>>c)||(row>>extra)||mode<0||mode>1)return 9;
  if(tag=="CSR_STATUS"){if(status_seen[mode])return 9;status_seen[mode]=true;if(a!=(mode==0?3:0)||b!=(mode==0?0:3)||c!=(mode==0?3:0)){std::cerr<<"CSR status/eligibility disagreement\n";return 8;}}
  else if(tag=="CSR_FACTOR"){
   if(a<0||a>2||seen[mode][a]){return 9;}
   seen[mode][a]=true;++n;
   const double factor=std::bit_cast<double>(static_cast<std::uint64_t>(b));const double logged=std::bit_cast<double>(static_cast<std::uint64_t>(c));
   if(!std::isfinite(factor)||!std::isfinite(logged))return 8;
   if(mode==0){if(factor!=7||logged!=8){std::cerr<<"CSR atomic refusal disagreement\n";return 8;}}
   else if(abs(B(logged)-logs[a])>B("3e-12")||abs(B(factor)-exp(logs[a]))>B("3e-12")){std::cerr<<"CSR independent median ratio disagreement\n";return 8;}
  }else return 9;
 }
 if(!complete||n!=6||!status_seen[0]||!status_seen[1])return 9;
 std::cout<<"CSR_REFERENCE_PASS factors=6 modes=2\n";return 0;
}
int verify_guards(const char* path){
 std::ifstream in(path);if(!in)return 2;std::string line,tag,extra;
 const I codes[9]={3,4,2,2,2,2,2,0,0};
 for(I k=0;k<9;++k){
  if(!std::getline(in,line))return 9;
  std::istringstream row(line);I index,status,normal;
  if(!(row>>tag>>index>>status>>normal)||tag!="CSR_GUARD"||(row>>extra)||index!=k)return 9;
  const I expected_normal=k<7?1:(k==7?5:3);
  if(status!=codes[k]||normal!=expected_normal){std::cerr<<"CSR guard status disagreement "<<k<<'\n';return 8;}
 }
 if(!std::getline(in,line)||line!="CSR_GUARDS_PASS")return 9;
 while(std::getline(in,line)){if(!line.empty())return 9;}
 std::cout<<"CSR_GUARD_REFERENCE_PASS rows=9\n";return 0;
}
int main(int argc,char** argv){try{if(argc==3&&std::string(argv[1])=="--csr-guards")return verify_guards(argv[2]);if(argc==3&&std::string(argv[1])=="--csr")return verify_csr(argv[2]);if(argc==2&&std::string(argv[1])=="--generate-named"){generate();return 0;}if(argc==3&&std::string(argv[1])=="--named")return verify(argv[2]);return 64;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

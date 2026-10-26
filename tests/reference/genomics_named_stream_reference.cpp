// C++23 framing/transport stress oracle; this is not a biological calibration.
#include <cstdint>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using I=std::int64_t;
void generate(I n,bool into){
 if(n!=100000&&n!=1000000)throw std::runtime_error("declared benchmark size");
 std::cout<<"use collections::integer_buffer::{integer_buffer_new,integer_buffer_at,integer_buffer_close}\nuse genomics::io::named_count_tsv::{named_count_tsv_new,named_count_tsv_ack,";
 if(into)std::cout<<"named_count_tsv_feed_slot,named_count_tsv_feed_into,named_count_tsv_finish_into";else std::cout<<"named_count_tsv_feed,named_count_tsv_finish";
 std::cout<<"}\nfn main()->i32 with Alloc,Mut,Div,Panic,IO {\n let n:i64="<<n<<"\n";
 std::cout<<R"SIO( var state=named_count_tsv_new(2,8)
 var id=integer_buffer_new(8,64);var counts=integer_buffer_new(2,16)
 let header:[i64;6]=[103,9,65,9,66,10]
 let body:[i64;6]=[120,9,49,9,50,10]
 var chunk:[i64;256]=[0;256]
 var position:i64=0;var headers:i64=0;var rows:i64=0;var total:i64=0;var id_hash:i64=0
 var done:bool=false
)SIO";
 if(into)std::cout<<" var result=named_count_tsv_feed_slot()\n";
 std::cout<<R"SIO( while !done {
  var length:i64=0
  while length<6 && position+length<6+6*n {
   let p=position+length
   chunk[length as usize]=if p<6 {header[p as usize]}else{body[((p-6)%6) as usize]}
   length=length+1
  }
)SIO";
 if(into)std::cout<<"  let code:i64=if length>0 {named_count_tsv_feed_into(&!state,&chunk,length,&!id,&!counts,&!result)}else{named_count_tsv_finish_into(&!state,&id,&!counts,&!result)}\n  assert(code==0)\n";
 else std::cout<<"  let result=if length>0 {named_count_tsv_feed(&!state,&chunk,length,&!id,&!counts)}else{named_count_tsv_finish(&!state,&id,&!counts)}\n";
 std::cout<<R"SIO(  assert(result.status==0)
  position=position+result.consumed
  if result.kind==1 {assert(result.index==headers && result.id_len==1);headers=headers+1}
  if result.kind!=0 {
   var j:i64=0
   while j<result.id_len {id_hash=(id_hash*131+integer_buffer_at(&id,j))%1000000007;j=j+1}
   id_hash=(id_hash*131)%1000000007
   if result.kind==2 {
    assert(result.index==rows && result.id_len==1 && integer_buffer_at(&id,0)==120)
    let a=integer_buffer_at(&counts,0);let b=integer_buffer_at(&counts,1)
    assert(a==1 && b==2);total=total+a+b;rows=rows+1
   }
   assert(named_count_tsv_ack(&!state))
  }else if result.finished {done=true}
 }
 println("NAMED_BENCH {} {} {} {} {} {}",n,rows,position,headers,total,id_hash)
 integer_buffer_close(id);integer_buffer_close(counts)
 println("NAMED_BENCH_PASS")
 0
}
)SIO";
}
I mix(I h,I value){return (h*131+value)%1000000007;}
int verify(I n,const char* path){
 if(n!=100000&&n!=1000000)return 64;
 std::ifstream in(path);if(!in)return 2;std::string tag,marker,extra;I declared,rows,position,headers,total,id_hash;
 if(!(in>>tag>>declared>>rows>>position>>headers>>total>>id_hash>>marker)||(in>>extra)||tag!="NAMED_BENCH"||marker!="NAMED_BENCH_PASS")return 9;
 I hash=0;hash=mix(mix(hash,65),0);hash=mix(mix(hash,66),0);for(I row=0;row<n;++row)hash=mix(mix(hash,120),0);
 if(declared!=n||rows!=n||position!=6+6*n||headers!=2||total!=3*n||id_hash!=hash){std::cerr<<"named stream count/identity/position disagreement\n";return 8;}
 std::cout<<"NAMED_STREAM_REFERENCE_PASS rows="<<n<<" bytes="<<position<<" headers=2\n";return 0;
}
std::string load(const char* path){std::ifstream in(path);if(!in)throw std::runtime_error("missing source");std::ostringstream s;s<<in.rdbuf();return s.str();}
void replace_one(std::string& text,const std::string& before,const std::string& after){const auto at=text.find(before);if(at==std::string::npos||text.find(before,at+1)!=std::string::npos)throw std::runtime_error("nonunique source adapter marker");text.replace(at,before.size(),after);}
void adapt(const char* path){
 auto text=load(path);
 replace_one(text,"named_count_tsv_feed,named_count_tsv_ack,named_count_tsv_finish","named_count_tsv_feed,named_count_tsv_ack,named_count_tsv_finish,named_count_tsv_feed_slot,named_count_tsv_feed_into,named_count_tsv_finish_into");
 // One event slot per generated case, outside its loop; the independent source
 // and expected cases remain supplied by genomics_count_identity_reference.
 const std::string before=" var code:i64=0;var error:i64=0;var done:bool=false";
 std::size_t at=0;int changed=0;while((at=text.find(before,at))!=std::string::npos){const auto after=" var event=named_count_tsv_feed_slot()\n"+before;text.replace(at,before.size(),after);at+=after.size();++changed;}
 if(changed!=15)throw std::runtime_error("source case count");
 const std::string call="let event=if n>0 {named_count_tsv_feed(&!state,&chunk,n,&!id,&!row)}else{named_count_tsv_finish(&!state,&id,&!row)}";
 const std::string replacement="let into_status:i64=if n>0 {named_count_tsv_feed_into(&!state,&chunk,n,&!id,&!row,&!event)}else{named_count_tsv_finish_into(&!state,&id,&!row,&!event)}\n  assert(into_status==event.status)";
 at=0;changed=0;while((at=text.find(call,at))!=std::string::npos){text.replace(at,call.size(),replacement);at+=replacement.size();++changed;}
 if(changed!=15){throw std::runtime_error("source call count");}
 std::cout<<text;
}
int report(const char* path){
 std::ifstream in(path);if(!in)return 2;
 struct R{std::string engine,variant;I size;double elapsed;I rss,exit;};
 std::vector<R> rows;R r;while(in>>r.engine>>r.size>>r.variant>>r.elapsed>>r.rss>>r.exit){if(r.exit!=0||r.elapsed<0||r.rss<=0)return 8;rows.push_back(r);}
 if(!in.eof())return 9;
 for(const auto& engine:{std::string("native"),std::string("seed")}){
  std::vector<double> before,after;I before_rss=0,after_rss=0;
  for(const auto& x:rows)if(x.engine==engine&&x.size==100000){
   if(x.variant=="before"){before.push_back(x.elapsed);before_rss=std::max(before_rss,x.rss);}
   else if(x.variant=="after"){after.push_back(x.elapsed);after_rss=std::max(after_rss,x.rss);}
   else return 8;
  }
  if(before.size()!=3||after.size()!=3)return 9;
  std::sort(before.begin(),before.end());std::sort(after.begin(),after.end());
  std::cout<<std::setprecision(10)<<"NAMED_RESOURCE "<<engine<<" rows=100000 before_median_seconds="<<before[1]<<" after_median_seconds="<<after[1]<<" before_max_rss_kib="<<before_rss<<" after_max_rss_kib="<<after_rss<<" rss_ratio="<<static_cast<double>(before_rss)/static_cast<double>(after_rss)<<'\n';
 }
 if(rows.size()!=12)return 9;
 std::cout<<"NAMED_RESOURCE_REPORT_PASS pairs=3_per_engine\n";return 0;
}
int main(int argc,char** argv){try{
 if(argc==3&&std::string(argv[1])=="--resources")return report(argv[2]);
 if(argc==4&&std::string(argv[1])=="--generate"){std::string mode=argv[3];if(mode!="legacy"&&mode!="into")return 64;generate(std::stoll(argv[2]),mode=="into");return 0;}
 if(argc==4&&std::string(argv[1])=="--verify")return verify(std::stoll(argv[2]),argv[3]);
 if(argc==3&&std::string(argv[1])=="--adapt-named"){adapt(argv[2]);return 0;}
 return 64;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

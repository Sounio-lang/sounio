// C++23 binary100 analytic controls for stated row-sparse count matrices.
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/multiprecision/cpp_dec_float.hpp>
using I=std::int64_t;using B=boost::multiprecision::cpp_dec_float_100;
void generate(bool sparse,I density){
 if(density!=1&&density!=10&&density!=100)throw std::runtime_error("density");
 std::cout<<R"SIO(use collections::integer_buffer::{integer_buffer_new,integer_buffer_set,integer_buffer_at,integer_buffer_close}
use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_set,numeric_buffer_at,numeric_buffer_close}
use genomics::dynamic_normalization::{dynamic_median_ratio_size_factors_workspace,dynamic_size_factors_slot,dynamic_size_factors_snapshot,dynamic_median_ratio_size_factors_into}
)SIO";
 if(sparse)std::cout<<"use genomics::count_csr::{count_csr_build,count_csr_status,count_csr_close,count_csr_size_factors_into}\n";
 std::cout<<R"SIO(fn main() with Alloc,Mut,Div,Panic,IO {
 let genes:i64=20000;let samples:i64=512
)SIO";
 std::cout<<" let density:i64="<<density<<"\n let active:i64=genes*density/100\n let nnz:i64=active*samples\n var checksum:i64=0\n";
 if(sparse){std::cout<<R"SIO( var offsets=integer_buffer_new(genes+1,(genes+1)*8)
 var columns=integer_buffer_new(nnz,nnz*8);var values=integer_buffer_new(nnz,nnz*8)
 var at:i64=0;var g:i64=0
 while g<genes {
  integer_buffer_set(&!offsets,g,at)
  if g<active {
   var s:i64=0
   while s<samples {
    let count=(g+1)*(s%7+1)
    integer_buffer_set(&!columns,at,s);integer_buffer_set(&!values,at,count)
    checksum=(checksum+count*((g*samples+s)%97+1))%1000000007
    at=at+1;s=s+1
   }
  }
  g=g+1
 }
 integer_buffer_set(&!offsets,genes,at)
 assert(at==nnz)
 let matrix=count_csr_build(genes,samples,nnz,offsets,columns,values,0)
 assert(count_csr_status(&matrix)==0)
 let input_bytes:i64=(genes+1+2*nnz)*8
)SIO";}
 else{std::cout<<R"SIO( var matrix=integer_buffer_new(genes*samples,genes*samples*8)
 var g:i64=0
 while g<genes {
  var s:i64=0
  while s<samples {
   let count=if g<active {(g+1)*(s%7+1)}else{0}
   integer_buffer_set(&!matrix,g*samples+s,count)
   checksum=(checksum+count*((g*samples+s)%97+1))%1000000007
   s=s+1
  }
  g=g+1
 }
 let input_bytes:i64=genes*samples*8
)SIO";}
 std::cout<<R"SIO( let plan=dynamic_median_ratio_size_factors_workspace(genes,samples)
 assert(plan.status==0)
 var work=numeric_buffer_new(plan.numeric_elements,plan.numeric_elements*8)
 var anchors=integer_buffer_new(genes,genes*8)
 var factors=numeric_buffer_new(samples,samples*8);var logs=numeric_buffer_new(samples,samples*8)
 var slot=dynamic_size_factors_slot()
 let scratch_bytes:i64=plan.payload_bytes+samples*16
)SIO";
 std::cout<<" println(\"COUNT_SCALE_INPUT {} {} {} {} {} {}\",density,active,nnz,checksum,input_bytes,scratch_bytes)\n var mode:i64=0\n while mode<2 {\n";
 if(sparse)std::cout<<"  let code=count_csr_size_factors_into(&matrix,mode,&!factors,&!logs,&!work,&!anchors,&!slot)\n";
 else std::cout<<"  let code=dynamic_median_ratio_size_factors_into(&matrix,genes,samples,samples,1,0,mode,&!factors,&!logs,&!work,&!anchors,&!slot)\n";
 std::cout<<R"SIO(  let fit=dynamic_size_factors_snapshot(&slot)
  assert(code==0 && fit.genes_used==active && fit.genes_excluded==genes-active)
  var s:i64=0
  while s<samples {
   let fb:i64=f64_to_bits(numeric_buffer_at(&factors,s));let lb:i64=f64_to_bits(numeric_buffer_at(&logs,s))
   println("COUNT_SCALE_FACTOR {} {} {} {}",mode,s,fb,lb);s=s+1
  }
  mode=mode+1
 }
)SIO";
 if(sparse)std::cout<<" count_csr_close(matrix)\n";else std::cout<<" integer_buffer_close(matrix)\n";
 std::cout<<R"SIO( numeric_buffer_close(work);integer_buffer_close(anchors);numeric_buffer_close(factors);numeric_buffer_close(logs)
 println("COUNT_SCALE_COMPLETE")
}
)SIO";
}
int verify(const char* path){
 std::ifstream in(path);if(!in)return 2;std::string line,tag,extra;I density,active,nnz,checksum,input_bytes,scratch_bytes;
 if(!std::getline(in,line)){return 9;}
 std::istringstream header(line);
 if(!(header>>tag>>density>>active>>nnz>>checksum>>input_bytes>>scratch_bytes)||tag!="COUNT_SCALE_INPUT"||(header>>extra))return 9;
 if((density!=1&&density!=10&&density!=100)||active!=20000*density/100||nnz!=active*512||scratch_bytes!=(3*20000+4*512)*8)return 8;
 if(input_bytes!=20000*512*8&&input_bytes!=(20000+1+2*nnz)*8)return 8;
 I wanted=0;for(I g=0;g<active;++g)for(I s=0;s<512;++s)wanted=(wanted+(g+1)*(s%7+1)*((g*512+s)%97+1))%1000000007;
 if(checksum!=wanted){std::cerr<<"scale exact input count checksum\n";return 8;}
 B center=0;for(I s=0;s<512;++s)center+=log(B(s%7+1));center/=512;
 I rows=0;bool complete=false;
 while(std::getline(in,line)){
  if(line=="COUNT_SCALE_COMPLETE"){if(complete)return 9;complete=true;continue;}
  std::istringstream row(line);I mode,s,fb,lb;if(!(row>>tag>>mode>>s>>fb>>lb)||tag!="COUNT_SCALE_FACTOR"||(row>>extra)||mode!=rows/512||s!=rows%512||rows>=1024)return 9;
  const auto factor=std::bit_cast<double>(static_cast<std::uint64_t>(fb));const auto logged=std::bit_cast<double>(static_cast<std::uint64_t>(lb));const B expected_log=log(B(s%7+1))-center;
  if(!std::isfinite(factor)||!std::isfinite(logged)||abs(B(logged)-expected_log)>B("3e-12")||abs(B(factor)-exp(expected_log))>abs(exp(expected_log))*B("3e-12")){std::cerr<<"scale analytic factor/log disagreement "<<mode<<' '<<s<<'\n';return 8;}++rows;
 }
 if(!complete||rows!=1024){return 9;}
 std::cout<<"COUNT_SCALE_REFERENCE_PASS density="<<density<<" rows=20000 samples=512 factors="<<rows<<" input_bytes="<<input_bytes<<'\n';return 0;
}
int main(int argc,char** argv){try{if(argc==4&&std::string(argv[1])=="--generate"){std::string mode=argv[2];if(mode!="dense"&&mode!="csr")return 64;generate(mode=="csr",std::stoll(argv[3]));return 0;}if(argc==2)return verify(argv[1]);return 64;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

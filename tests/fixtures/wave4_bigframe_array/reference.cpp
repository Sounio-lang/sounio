#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
static std::string bits(double v){return std::to_string(std::bit_cast<std::int64_t>(v));}
static std::string expected(const std::string& name){
 const std::int64_t word6=11+2*22+3*33+4*44+5*55+6*66;
 const std::int64_t word7=word6+7*707;
 const double mixed6=11.0+2.0*22.0+3.0*33.0+4.0*44.0+5.0*1.25+6.0*2.5;
 const double frame6=1.0+2.0*22.0+3.0*33.0+4.0*44.0+5.0*1.25+6.0*2.5;
 std::string value;
 if(name=="word6")value=std::to_string(word6);
 else if(name=="word7"||name=="word_wrap5")value=std::to_string(word7);
 else if(name=="mixed6")value=bits(mixed6);
 else if(name=="mixed7"||name=="mixed_wrap5")value=bits(mixed6+7.0*707.0);
 else if(name=="frame6")value=bits(frame6);
 else if(name=="frame7"||name=="frame_wrap5")value=bits(frame6+7.0*707.0);
 else if(name=="array")value=bits(375.0/100.0);
 else if(name=="scaled_raw"||name=="scaled_index"){
  // Median of the five exact scaled integers, then divide by the exact scale.
  const std::int64_t sorted[5]={125,250,375,500,625};
  const double median=static_cast<double>(sorted[2])/100.0;
  return "BEFORE_SCALED_MEDIAN\nAFTER_SCALED_MEDIAN\nSCALED_VALUE_BITS="+bits(median)+"\nSCALED_MEDIAN_COMPONENT_OK\n";
 }else throw std::runtime_error("unknown producer");
 return "BEGIN\n"+value+"\nEND\n";
}
int main(int argc,char** argv){
 try{
  if(argc==4&&std::string(argv[1])=="--index-variant"){
   std::ifstream in(argv[2],std::ios::binary);if(!in)throw std::runtime_error("missing library");
   std::string src{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
   const auto begin=src.find("fn bf_group_quantile_scaled(");if(begin==std::string::npos)throw std::runtime_error("missing function");
   const auto end=src.find("\n}\n",begin);
   const std::string old="read_f64(res as *mut f64, k)";
   const auto pos=src.find(old,begin);
   if(end==std::string::npos||pos==std::string::npos||pos>=end||src.find(old,pos+old.size())<end)throw std::runtime_error("nonunique expression");
   src.replace(pos,old.size(),"res[k]");
   std::ofstream out(argv[3],std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot write variant");out<<src;if(!out)throw std::runtime_error("short variant write");
   std::cout<<"INDEX_VARIANT_ONE_EXPRESSION_ONLY\n";return 0;
  }
  if(argc==2&&std::string(argv[1])=="--self-test"){
   const auto good=expected("word7");auto wrong=good;wrong[6]=wrong[6]=='9'?'8':'9';
   if(wrong==good||good.substr(0,good.size()-4)==good||expected("mixed7")==good)throw std::runtime_error("ineffective control");
   std::cout<<"ABI_ORACLE_THREE_WRONG_OUTPUT_CONTROLS_OK\n";return 0;
  }
  if(argc==3&&std::string(argv[1])=="--expected"){std::cout<<expected(argv[2]);return 0;}
  if(argc==4&&std::string(argv[1])=="--verify"){
   std::ifstream in(argv[3],std::ios::binary);if(!in)throw std::runtime_error("missing output");
   const std::string actual{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
   if(actual!=expected(argv[2])){std::cerr<<"OUTPUT_REFUSED "<<argv[2]<<"\n";return 1;}
   std::cout<<"ABI_VALUE_BITS_OK "<<argv[2]<<"\n";return 0;
  }
  throw std::runtime_error("usage");
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}

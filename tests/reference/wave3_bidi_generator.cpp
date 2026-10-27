#include "wave3_bidi_cases.hpp"
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){
 try{
  using namespace bidi_cases;const bool mutant=argc==4&&std::string(argv[1])=="--wrong-level-control";
  need(argc==3||mutant,"usage: bidi-generator [--wrong-level-control] UCD_DIRECTORY FRESH_OUTPUT");
  const std::filesystem::path root=argv[mutant?2:1],out=argv[mutant?3:2];need(!std::filesystem::exists(out),"refuse existing output");std::filesystem::create_directories(out);
  if(mutant){
   Case chosen;bool found=false;
   read(root,[&](const Case& c){if(!found&&c.kind==0&&c.mode==2&&c.cps==std::vector<U>{0x660}&&c.levels==std::vector<int>{2}&&c.order==std::vector<U>{0}){chosen=c;found=true;}});
   need(found,"missing official lone-AN autoLTR witness");chosen.levels[0]=3;
   std::ofstream binary(out/"wrong-level.bin",std::ios::binary);binary.write("SBIDI18\0",8);put(binary,1);put(binary,1);write(binary,chosen);binary.close();need(bool(binary),"write mutant");
   std::cout<<"BIDI_WRONG_LEVEL_CONTROL_GENERATED official_level=2 mutant_level=3 visual_order_unchanged=true\n";return 0;
  }
  std::ofstream manifest(out/"CHUNKS.tsv"),report(out/"COUNTS.json");std::vector<Case> chunk,smoke;std::array<U,2> smoke_counts{};
  std::uint64_t bytes=16,total=0,points=0,max_n=0;std::array<std::uint64_t,2> kinds{};U index=0;
  auto flush=[&]{
   if(chunk.empty())return;
   std::ostringstream name;name<<std::setw(6)<<std::setfill('0')<<index++<<".bin";
   std::ofstream binary(out/name.str(),std::ios::binary);binary.write("SBIDI18\0",8);put(binary,1);put(binary,static_cast<U>(chunk.size()));
   for(const auto& c:chunk){write(binary,c);}
   binary.close();need(bool(binary),"close corpus chunk");
   manifest<<name.str()<<'\t'<<chunk.size()<<'\t'<<bytes<<'\n';chunk.clear();bytes=16;
  };
  read(root,[&](const Case& c){
   if(smoke_counts[c.kind]<32&&(c.kind==1||std::any_of(c.levels.begin(),c.levels.end(),[](int x){return x>=0;}))){smoke.push_back(c);++smoke_counts[c.kind];}
   const auto size=encoded_size(c);need(size+16<=1048576,"single official case exceeds chunk budget");
   if(chunk.size()>=4096||bytes+size>1048576)flush();
   chunk.push_back(c);bytes+=size;++total;++kinds[c.kind];points+=c.cps.size();max_n=std::max(max_n,static_cast<std::uint64_t>(c.cps.size()));
  });flush();need(total&&kinds[0]&&kinds[1],"empty corpus selection");
  need(smoke_counts[0]==32&&smoke_counts[1]==32,"empty smoke profile");
  std::ofstream smoke_file(out/"SMOKE.bin",std::ios::binary);smoke_file.write("SBIDI18\0",8);put(smoke_file,1);put(smoke_file,static_cast<U>(smoke.size()));
  for(const auto& c:smoke){write(smoke_file,c);}
  smoke_file.close();need(bool(smoke_file),"close smoke corpus");
  report<<"{\"unicode\":\"18.0.0\",\"cases\":"<<total<<",\"class_cases\":"<<kinds[0]<<",\"character_cases\":"<<kinds[1]<<",\"scalars\":"<<points<<",\"max_case_scalars\":"<<max_n<<",\"chunks\":"<<index<<"}\n";
  need(bool(report)&&bool(manifest),"write reports");
  std::cout<<"BIDI18_CORPUS_GENERATED cases="<<total<<" class_cases="<<kinds[0]<<" character_cases="<<kinds[1]<<" chunks="<<index<<"\n";
 }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}

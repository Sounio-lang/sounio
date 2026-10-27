#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
#include <stdexcept>
static void put16(std::vector<unsigned char>&b,std::size_t p,std::uint16_t x){b.at(p)=static_cast<unsigned char>(x);b.at(p+1)=static_cast<unsigned char>(x>>8);}
int main(int argc,char**argv){
 try{
  if(argc!=4){throw std::runtime_error("INPUT_A ATLAS_OUTPUT CONFIG_OUTPUT required");}
  std::ifstream f(argv[1],std::ios::binary);
  std::vector<unsigned char>b((std::istreambuf_iterator<char>(f)),{});
  if(b.size()!=384||b[0]!=92||b[1]!=1||b[2]!=0||b[3]!=0){throw std::runtime_error("expected pinned little-endian NIfTI1 fixture");}
  b.resize(352);put16(b,40,3);put16(b,48,1);put16(b,68,1002);
  const unsigned char labels[]{0,1,1,2,2,3,0,0};b.insert(b.end(),labels,labels+8);
  std::ofstream atlas(argv[2],std::ios::binary);
  atlas.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size()));
  std::ofstream cfg(argv[3]);
  cfg<<"formula\t1\ncoverage_mode\t1\nexpected_regions\t4\nalignment_policy\t1\n"
     <<"fd_mean\t1.0\t1.0\ndvars\t1.0\t10.0\ninverse_tsnr\t1.0\t1.0\nfc_ci_width\t1.0\t2.0\nuncovered_fraction\t1.0\t1.0\n";
  if(!atlas||!cfg){throw std::runtime_error("output write");}
  std::cout<<"FMRI_ATLAS_FIXTURES_WRITTEN voxels=8 labels=3 declared_regions=4\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

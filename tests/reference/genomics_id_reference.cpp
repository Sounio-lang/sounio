// C++23 independent byte-exact sorting/join oracle. No Sounio sorting code reused.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
using I=std::int64_t;
bool less_bytes(const std::string&a,const std::string&b){
 return std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end(),[](char x,char y){return static_cast<unsigned char>(x)<static_cast<unsigned char>(y);});
}
std::vector<std::string> data(int c){
 if(c==0)return{};
 if(c==3){std::vector<std::string> out;for(int b=1;b<=255;++b)if(b!=9&&b!=10&&b!=13)out.emplace_back(1,static_cast<char>(b));return out;}
 if(c==1)return{"geneB","geneA","geneB"," GeneA","genea","\xc3\xa9","e\xcc\x81","with space","GeneA"};
 std::vector<std::string> out;constexpr char hex[]="0123456789abcdef";
 for(int i=4096;i>=0;--i){std::string s="gene0000";for(int j=0;j<4;++j)s[7-j]=hex[(i>>(4*j))&15];out.push_back(s);}return out;
}
bool row(std::ifstream&in,std::vector<std::string>&v){std::string s;if(!std::getline(in,s))return false;std::istringstream p(s);std::string t;v.clear();while(p>>t)v.push_back(t);return true;}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ifstream in(argv[1]);if(!in)return 3;
 std::vector<std::string> v;I checked=0,bytes=0;
 for(int c=0;c<4;++c){
  const auto names=data(c);const I n=I(names.size());std::vector<I> sorted(n),groups(n);
  std::iota(sorted.begin(),sorted.end(),I(0));
  std::stable_sort(sorted.begin(),sorted.end(),[&](I a,I b){return less_bytes(names[a],names[b]);});
  I unique=0;std::string prev;
  for(I i=0;i<n;++i){if(i==0||names[sorted[i]]!=prev){++unique;prev=names[sorted[i]];}groups[sorted[i]]=unique-1;}
  if(!row(in,v)||v.size()!=6||v[0]!="ID_CASE"||std::stoll(v[1])!=c||std::stoll(v[2])!=0||std::stoll(v[3])!=n||std::stoll(v[4])!=unique)return 4;
  const I compared=std::stoll(v[5]);if(compared<0||compared>(c==2?4000000:(c==3?100000:10000))||(n>1&&compared==0))return 5;
  for(I i=0;i<n;++i){
   if(!row(in,v)||v.size()!=6||v[0]!="ID_ROW"||std::stoll(v[1])!=c||std::stoll(v[2])!=i)return 4;
   if(std::stoll(v[3])!=groups[i]||std::stoll(v[4])!=sorted[i]||std::stoll(v[5])!=I(names[i].size())){std::cerr<<"identity row/group/order "<<c<<" "<<i<<"\n";return 8;}
   for(I j=0;j<I(names[i].size());++j){
    if(!row(in,v)||v.size()!=5||v[0]!="ID_BYTE"||std::stoll(v[1])!=c||std::stoll(v[2])!=i||std::stoll(v[3])!=j)return 4;
    if(std::stoll(v[4])!=static_cast<unsigned char>(names[i][j])){std::cerr<<"opaque identity bytes "<<c<<" "<<i<<" "<<j<<"\n";return 8;}++bytes;
   }++checked;
  }
 }
 const std::vector<std::string> left{"geneC","geneA","geneB"},right{"geneB","geneC","geneA"},missing{"geneC","geneA","geneZ"};
 for(const std::string&kind:{std::string("ID_JOIN"),std::string("ID_MISSING")}){
  const auto& source=kind=="ID_JOIN"?left:missing;
  for(I i=0;i<3;++i){
   const auto found=std::find(right.begin(),right.end(),source[i]);const I expected=found==right.end()?-1:I(found-right.begin());
   if(!row(in,v)||v.size()!=3||v[0]!=kind||std::stoll(v[1])!=i)return 4;
   if(std::stoll(v[2])!=expected){std::cerr<<"identity join original index\n";return 8;}
  }
 }
 std::string line;
 if(!std::getline(in,line)||line!="ID_REFUSALS duplicate3 budget4 byte2 empty2 join_missing6 join_budget4 ambiguous5")return 6;
 if(!std::getline(in,line)||line!="COUNT_IDENTITIES_PASS"||std::getline(in,line)){std::cerr<<"identity completion\n";return 9;}
 std::cout<<"GENOMICS_ID_REFERENCE_PASS identities="<<checked<<" opaque_bytes="<<bytes<<" sort=independent_stable_sort joins=original_indices\n";
}

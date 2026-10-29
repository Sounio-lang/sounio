#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <regex>
#include <stdexcept>
std::string read(const char*p){std::ifstream f(p);if(!f)throw std::runtime_error(p);return {std::istreambuf_iterator<char>(f),{}};}
std::vector<std::string> en(const std::string&s,const std::string&n){std::smatch m;if(!std::regex_search(s,m,std::regex("pub enum "+n+R"(\s*\{([^}]+)\})")))throw std::runtime_error("enum");std::vector<std::string>v;std::string t=m[1];std::regex r(R"(([A-Za-z][A-Za-z0-9_]*)\s*,)");for(auto i=std::sregex_iterator(t.begin(),t.end(),r);i!=std::sregex_iterator();++i)v.push_back((*i)[1]);return v;}
int main(int ac,char**av){try{if(ac!=4)throw std::runtime_error("argc");auto k=en(read(av[1]),"ItemKind"),s=en(read(av[2]),"SymbolKind");if(k.size()!=28||s.size()!=25)throw std::runtime_error("enum census");std::map<std::string,std::string>m={
{"ItemFn","SymFunction"},{"ItemStruct","SymStruct"},{"ItemEnum","SymEnum"},{"ItemTypeAlias","SymTypeAlias"},{"ItemModels","SymTypeAlias"},{"ItemPolicy","SymPolicy"},{"ItemAcquisitionPolicy","SymAcquisitionPolicy"},{"ItemRecoursePolicy","SymRecoursePolicy"},{"ItemTransitionPolicy","SymTransitionPolicy"},{"ItemTransitionProtocol","SymTransitionProtocol"},{"ItemMonitoringPolicy","SymMonitoringPolicy"},{"ItemDecisionPolicy","SymDecisionPolicy"},{"ItemDeferralPolicy","SymDeferralPolicy"},{"ItemAlternativePolicy","SymAlternativePolicy"},{"ItemAlternativeFrontier","SymAlternativeFrontier"},{"ItemValidation","SymPolicy"},{"ItemAudit","SymAudit"},{"ItemEffect","SymEffect"},{"ItemSession","SymTypeAlias"},{"ItemTrait","SymTypeAlias"},{"ItemAlgebra","SymTypeAlias"},{"ItemStudy","SymTypeAlias"},{"ItemOntology","SymTypeAlias"}};
std::ifstream f(av[3]);std::string line;int rows=0;while(std::getline(f,line)){std::istringstream z(line);long long i,p,b,c,d,kind,first,len,drop;std::string extra;if(!(z>>i>>p>>b>>c>>d>>kind>>first>>len>>drop)||(z>>extra))throw std::runtime_error("row shape");if(i!=rows/2||p!=rows%2||i<0||i>=28)throw std::runtime_error("row order");bool exported=p==1&&m.contains(k[i]);long long expected=-1;if(exported){auto it=std::find(s.begin(),s.end(),m.at(k[i]));if(it==s.end())throw std::runtime_error("symbol");expected=it-s.begin();}if(b!=0||c!=exported||d!=exported||kind!=expected||first!=(exported?120:-1)||len!=(exported?1:-1)||drop!=0)throw std::runtime_error("export mismatch at "+std::to_string(rows));++rows;}
if(rows!=56)throw std::runtime_error("rows");std::cout<<"PASS actual_ItemKind=28 private_public=56 exact_exports=56\n";return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}


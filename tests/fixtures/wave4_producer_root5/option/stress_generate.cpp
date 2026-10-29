#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <tuple>
#include <string>
int main(int ac,char**av){if(ac!=2)return 2;namespace fs=std::filesystem;fs::path p=av[1];fs::create_directories(p);{
std::ofstream f(p/"many_functions.sio");for(int i=0;i<5000;++i)f<<"fn f"<<i<<"(x:Option<i64>)->(Option<i64>,i64){match x{Some(v)=>(Some(v),1),None=>(None,0)}}\n";f<<"fn un(x:Option<i64>)->i64{match x{Some(v)=>v,None=>-9}}\nfn main()->i32 with IO{let a=f0(Some(17));let b=f4999(None);print(un(a.0));print(\" \");println(un(b.0));0}\n";}
for(auto [name,count,coerce]:{std::tuple{"proof_budget_near",1022,true},std::tuple{"proof_budget_closed",1023,true},std::tuple{"proof_budget_unused",1100,false}}){std::ofstream f(p/(std::string(name)+".sio"));f<<"fn f(x:Option<i64>)";if(coerce)f<<"->(Option<i64>,i64)";else f<<"->i64";f<<"{\n";for(int i=0;i<count;++i)f<<"(Some(1),0);\n";if(coerce)f<<"match x{Some(v)=>(Some(v),1),None=>(None,0)}\n";else f<<"17\n";f<<"}\nfn main()->i32 with IO{";if(coerce)f<<"let a=f(Some(17));println(a.1);";else f<<"println(f(None));";f<<"0}\n";}return 0;}


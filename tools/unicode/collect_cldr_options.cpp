// Complete CLDR option-matrix collector. Accounting only, not a collation oracle.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
[[noreturn]]void error(const std::string&s){throw std::runtime_error(s);}
std::string read(const fs::path&p){std::ifstream in(p,std::ios::binary);std::ostringstream out;out<<in.rdbuf();if(!in)error("read "+p.string());return out.str();}
std::uint64_t number(const std::string&s){std::size_t end=0;auto v=std::stoull(s,&end);if(end!=s.size())error("integer syntax");return v;}
struct Entry{std::string name;std::uint64_t samples;};
struct Run{std::vector<Entry> entries;std::uint64_t comparisons=0,count=0;};
Run parse(const fs::path&p){Run out;std::istringstream in(read(p));std::string line;bool footer=false;while(std::getline(in,line)){if(line.empty())continue;if(line.starts_with("CLDR_CE_GRAPH_ORACLE_ENTRY_OK ")){if(footer)error("entry after footer");std::istringstream fields(line);std::string marker,name,samples,extra;fields>>marker>>name>>samples;if(fields>>extra||name.empty()||!samples.starts_with("samples="))error("entry schema");auto n=number(samples.substr(8));if(n<2)error("empty entry samples");out.entries.push_back({name,n});}else if(line.starts_with("CLDR_CE_GRAPH_ORACLE_OK ")){if(footer)error("duplicate footer");footer=true;std::istringstream fields(line);std::string marker,entries,comparisons,extra;fields>>marker>>entries>>comparisons;if(fields>>extra||!entries.starts_with("entries=")||!comparisons.starts_with("comparisons="))error("footer schema");out.count=number(entries.substr(8));out.comparisons=number(comparisons.substr(12));}else error("unknown oracle output line");}if(!footer||out.entries.empty()||out.count!=out.entries.size()||!out.comparisons)error("missing/empty/inconsistent footer");return out;}
void zero(const fs::path&p){auto value=read(p);if(value!="0\n")error("nonzero status "+p.string());}
std::size_t lines(const std::string&s){std::istringstream in(s);std::string line;std::size_t n=0;while(std::getline(in,line))if(!line.empty())++n;return n;}
void verified(const fs::path&p,std::size_t expected=0){std::istringstream in(read(p));std::string line;std::size_t n=0;while(std::getline(in,line)){if(line.empty())continue;if(!line.ends_with(": OK"))error("failed guard "+p.string());++n;}if(!n||(expected&&n!=expected))error("empty/incomplete guard "+p.string());}
int main(int argc,char**argv){try{if(argc<6)return 64;auto baseline=parse(argv[1]);auto expected_sha=std::string(argv[2]);if(expected_sha.size()!=64)error("expected oracle SHA width");fs::path output=argv[3];auto parts=static_cast<std::size_t>(argc-4);if(parts>baseline.entries.size())error("empty partition");std::map<std::string,std::size_t> expected;std::uint64_t base_comparisons=0;for(std::size_t i=0;i<baseline.entries.size();++i){const auto&e=baseline.entries[i];if(!expected.emplace(e.name,i).second)error("duplicate baseline entry");base_comparisons+=e.samples-1;}if(base_comparisons!=baseline.comparisons)error("baseline accounting");constexpr std::uint64_t configs=5*2*2*2*3*4;std::vector<bool> seen(baseline.entries.size());std::uint64_t total=0;
 for(std::size_t part=0;part<parts;++part){fs::path dir=argv[part+4];zero(dir/"WORKER_STATUS");zero(dir/"ORACLE_STATUS");if(!read(dir/"ORACLE.sha256").starts_with(expected_sha+"  "))error("oracle identity mismatch");verified(dir/"ORACLE.before.verify",1);verified(dir/"ORACLE.final.verify",1);auto source_before=read(dir/"SOURCE.before.verify"),source_after=read(dir/"SOURCE.final.verify");if(source_before!=source_after)error("source guard disagreement");verified(dir/"SOURCE.final.verify");verified(dir/"UPSTREAM.final.verify",lines(read(dir/"UPSTREAM.sha256")));auto run=parse(dir/"oracle.stdout");std::uint64_t count=0;for(const auto&entry:run.entries){auto found=expected.find(entry.name);if(found==expected.end())error("unknown locale entry");auto index=found->second;if(index%parts!=part||seen[index])error("wrong/duplicate partition locale");if(entry.samples!=baseline.entries[index].samples)error("changed sample domain");seen[index]=true;count+=(entry.samples-1)*configs;}if(count!=run.comparisons)error("option configuration accounting");total+=count;}
 for(bool value:seen){if(!value)error("missing locale entry");}
 if(total!=base_comparisons*configs){error("complete comparison accounting");}
 std::ofstream out(output);out<<"entries\t"<<baseline.entries.size()<<"\ncomparisons\t"<<total<<"\nconfigurations\t"<<configs<<"\npartitions\t"<<parts<<"\noracle_sha256\t"<<expected_sha<<"\n";if(!out){error("summary output");}
 std::cout<<"CLDR_OPTIONS_COMPLETE_OK entries="<<baseline.entries.size()<<" comparisons="<<total<<" configurations="<<configs<<" parts="<<parts<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"CLDR_OPTIONS_COLLECT_ERROR "<<e.what()<<"\n";return 1;}}

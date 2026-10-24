#include <sys/xattr.h>
#include <sys/statvfs.h>
#include <filesystem>
#include <charconv>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <cstdint>
namespace fs=std::filesystem;
std::string read(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("open "+p);return {std::istreambuf_iterator<char>(f),{}};}
std::uint64_t number(const std::string&s){std::uint64_t v=0;auto r=std::from_chars(s.data(),s.data()+s.size(),v);if(r.ec!=std::errc{}||r.ptr!=s.data()+s.size())throw std::runtime_error("number");return v;}
std::uint64_t attr(const char*k){char b[128];auto n=getxattr("/var/lib/cephfs-slurm/training",k,b,sizeof b);if(n<0)throw std::runtime_error("xattr");return number({b,static_cast<std::size_t>(n)});}
int main(){try{
const std::string a="/home/devsounio/sounio-artifact-archive/2026-10-20/stats-terminal-expanded";
if(read(a+"/OFFLOAD_COMPLETE")!="ALL_107_SOURCES_OFFLOADED\n")throw std::runtime_error("marker");
auto journal=read(a+"/OFFLOAD_JOURNAL.tsv");if(journal!=read("/tmp/expanded107-B-journal.tsv"))throw std::runtime_error("journal A/B differ");
std::map<std::string,std::string> expected;std::uint64_t bytes=0;std::istringstream plan(read(a+"/pilot.tsv"));std::string line;std::getline(plan,line);while(std::getline(plan,line)){std::istringstream s(line);std::string name,size,archive,sha;std::getline(s,name,'\t');std::getline(s,size,'\t');std::getline(s,archive,'\t');std::getline(s,sha,'\t');if(!expected.emplace(name,sha).second)throw std::runtime_error("duplicate plan");bytes+=number(size);}
std::set<std::string> cohort;std::istringstream names(read(a+"/COHORT_RUNS.txt"));while(std::getline(names,line))cohort.insert(line);
if(expected.size()!=107||cohort.size()!=107||bytes!=1240532571ULL)throw std::runtime_error("cohort size/bytes");
std::size_t absent=0;for(auto&[name,sha]:expected){if(!cohort.contains(name))throw std::runtime_error("cohort mismatch");auto path="/var/lib/cephfs-slurm/training/sounio-foundry/runs/"+name+"/source.tar.gz";std::error_code ec;auto st=fs::symlink_status(path,ec);if(st.type()==fs::file_type::not_found)++absent;else throw std::runtime_error("source exists or inaccessible "+name);}
std::map<std::string,unsigned> phases;std::istringstream rows(journal);std::getline(rows,line);std::size_t intents=0,removed=0;while(std::getline(rows,line)){std::istringstream s(line);std::string phase,name,sha;std::getline(s,phase,'\t');std::getline(s,name,'\t');std::getline(s,sha,'\t');if(!expected.contains(name)||expected[name]!=sha)throw std::runtime_error("journal scope/hash");auto&bits=phases[name];if(phase=="intent"&&bits==0){bits=1;++intents;}else if(phase=="removed"&&bits==1){bits=3;++removed;}else throw std::runtime_error("journal order/duplicate");}
if(absent!=107||intents!=107||removed!=107)throw std::runtime_error("incomplete journal/absence");
auto used=attr("ceph.dir.rbytes"),quota=attr("ceph.quota.max_bytes");struct statvfs v{};if(statvfs("/var/lib/cephfs-slurm/training",&v))throw std::runtime_error("statvfs");
constexpr std::uint64_t earlier=22408313;
std::cout<<"source_absent\t"<<absent<<"\njournal_intent\t"<<intents<<"\njournal_removed\t"<<removed<<"\njournal_A_B_byte_identical\ttrue\nexpanded107_source_bytes\t"<<bytes<<"\nearlier_first_source_bytes\t"<<earlier<<"\ncombined_offloaded_source_bytes\t"<<bytes+earlier<<"\nquota_bytes\t"<<quota<<"\nused_logical_bytes\t"<<used<<"\nquota_headroom_signed_bytes\t"<<static_cast<std::int64_t>(quota)-static_cast<std::int64_t>(used)<<"\nstatvfs_available_bytes\t"<<static_cast<std::uint64_t>(v.f_bavail)*v.f_frsize<<"\nPOST_OFFLOAD_AUDIT_PASS\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}


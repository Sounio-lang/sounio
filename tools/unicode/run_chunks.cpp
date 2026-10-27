// C++23 Slurm-native chunk runner. Each process gets a fresh Sounio arena.
// This orchestrates test fixtures; it does not implement production Unicode.
#include <array>
#include <algorithm>
#ifndef SOUNIO_CHUNK_KIND_MIN
#define SOUNIO_CHUNK_KIND_MIN 1
#endif
#ifndef SOUNIO_CHUNK_KIND_MAX
#define SOUNIO_CHUNK_KIND_MAX 4
#endif
#ifndef SOUNIO_CHUNK_NAME_PATTERN
#define SOUNIO_CHUNK_NAME_PATTERN "^(norm|segment|case|collation)-[0-9]{6}\\.bin$"
#endif
#ifndef SOUNIO_CHUNK_FILE_DESTINATION
#define SOUNIO_CHUNK_FILE_DESTINATION "tests/stdlib/wave3-core/data/current.bin"
#endif
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
namespace fs=std::filesystem;
using N=std::uint64_t;
[[noreturn]] void fail(const std::string& s){throw std::runtime_error(s);}
N number(const std::string& s){N n=0;auto[p,e]=std::from_chars(s.data(),s.data()+s.size(),n);if(e!=std::errc{}||p!=s.data()+s.size())fail("invalid integer: "+s);return n;}
std::uint32_t word(const std::array<unsigned char,24>& b,std::size_t at){std::uint32_t n=0;for(std::size_t i=0;i<4;++i)n|=std::uint32_t{b[at+i]}<<(8*i);return n;}
struct Fixture{std::string name;N kind,rows,bytes;};
std::vector<Fixture> manifest(const fs::path& root){
 std::ifstream in(root/"MANIFEST.tsv");if(!in)fail("fixture manifest missing");std::string line;
 if(!std::getline(in,line)||line!="filename\tkind\trows\tpayload_bytes")fail("manifest header");
 std::vector<Fixture> result;const std::regex safe(SOUNIO_CHUNK_NAME_PATTERN);
 while(std::getline(in,line)){
  if(std::count(line.begin(),line.end(),'\t')!=3)fail("manifest fields");
  std::istringstream fields(line);std::array<std::string,4> parts;for(auto& p:parts)if(!std::getline(fields,p,'\t'))fail("manifest fields");
  Fixture f{parts[0],number(parts[1]),number(parts[2]),number(parts[3])};
  if(!std::regex_match(f.name,safe)||f.kind<SOUNIO_CHUNK_KIND_MIN||f.kind>SOUNIO_CHUNK_KIND_MAX||f.rows==0||f.rows>512||f.bytes>1048576)fail("manifest value");
  if(fs::file_size(root/f.name)!=f.bytes+24)fail("fixture size mismatch: "+f.name);
  std::ifstream file(root/f.name,std::ios::binary);std::array<unsigned char,24> b{};file.read(reinterpret_cast<char*>(b.data()),b.size());
  if(!file||std::string(b.begin(),b.begin()+8)!="SOUNUTF1"||word(b,8)!=f.kind||word(b,12)!=f.rows||word(b,16)!=f.bytes||word(b,20)!=0x00120000)fail("fixture header mismatch: "+f.name);
  result.push_back(std::move(f));
 }
 if(result.empty())fail("empty fixture selection");return result;
}
int execute(const fs::path& executable,const fs::path& cwd){
 pid_t child=fork();if(child<0)fail("fork");
 if(child==0){
  setpgid(0,0);if(chdir(cwd.c_str())!=0)_exit(125);
  int out=open("native.stdout",O_WRONLY|O_CREAT|O_EXCL,0600),err=open("native.stderr",O_WRONLY|O_CREAT|O_EXCL,0600);
  if(out<0||err<0||dup2(out,STDOUT_FILENO)<0||dup2(err,STDERR_FILENO)<0)_exit(125);
  close(out);close(err);execl(executable.c_str(),executable.c_str(),static_cast<char*>(nullptr));_exit(127);
 }
 int status=0;auto deadline=std::chrono::steady_clock::now()+std::chrono::minutes(5);
 for(;;){
  pid_t waited=waitpid(child,&status,WNOHANG);if(waited<0)fail("waitpid");if(waited==child)break;
  if(std::chrono::steady_clock::now()>deadline){kill(-child,SIGKILL);kill(child,SIGKILL);waitpid(child,&status,0);return 124;}
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
 }
 if(WIFEXITED(status))return WEXITSTATUS(status);if(WIFSIGNALED(status))return 128+WTERMSIG(status);return 125;
}
std::pair<bool,N> observed(const fs::path& file){
 std::ifstream input(file);if(!input)fail("native stdout missing");std::string line;bool marker=false,have=false;N rows=0;
 while(std::getline(input,line)){
  if(line=="UNICODE_CONFORMANCE_OK")marker=true;
  N n=0;auto[p,e]=std::from_chars(line.data(),line.data()+line.size(),n);
  if(!line.empty()&&e==std::errc{}&&p==line.data()+line.size()){rows=n;have=true;}
 }
 return {marker&&have,rows};
}
#ifndef SOUNIO_CHUNK_RUNNER_NO_MAIN
int main(int argc,char**argv){
 try{
  if(argc!=6){std::cerr<<"usage: run_chunks ELF FIXTURE_DIRECTORY NEW_RUN_DIRECTORY PART PARTS\n";return 64;}
  fs::path executable=fs::canonical(argv[1]),fixtures=fs::canonical(argv[2]),out=fs::absolute(argv[3]);N part=number(argv[4]),parts=number(argv[5]);
  if(parts==0||parts>1024||part>=parts)fail("invalid partition");auto all=manifest(fixtures);
  std::vector<Fixture> selected;for(std::size_t i=0;i<all.size();++i)if(i%parts==part)selected.push_back(all[i]);if(selected.empty())fail("empty partition");
  if(!fs::create_directory(out))fail("run directory must be new");
  std::ofstream results(out/"RESULTS.tsv");results<<"fixture\tkind\texpected_rows\texit_status\tmarker\tactual_rows\n";N completed=0,rows=0;
  for(const auto& f:selected){
   auto cwd=out/fs::path(f.name).stem();auto destination=cwd/SOUNIO_CHUNK_FILE_DESTINATION;fs::create_directories(destination.parent_path());fs::copy_file(fixtures/f.name,destination);
   int status=execute(executable,cwd);auto [marker,actual]=observed(cwd/"native.stdout");
   results<<f.name<<'\t'<<f.kind<<'\t'<<f.rows<<'\t'<<status<<'\t'<<marker<<'\t'<<actual<<'\n';results.flush();if(!results)fail("result output");
   if(status!=0||!marker||actual!=f.rows){std::cerr<<"UNICODE_CHUNK_FAILURE "<<f.name<<" exit="<<status<<" expected="<<f.rows<<" actual="<<actual<<" marker="<<marker<<"\n";return 1;}
   ++completed;rows+=actual;
  }
  if(completed!=selected.size())fail("incomplete partition");
  std::ifstream scope(fixtures/"SCOPE.txt");std::ofstream copied(out/"FIXTURE_SCOPE.txt");copied<<scope.rdbuf();
  std::ofstream status(out/"STATUS");status<<"UNICODE_CHUNK_PART_OK\npart="<<part<<"\nparts="<<parts<<"\nfiles="<<completed<<"\nrows="<<rows<<"\n";
  if(!status||!copied)fail("scope/status output");std::cout<<"UNICODE_CHUNK_PART_OK part="<<part<<" parts="<<parts<<" files="<<completed<<" rows="<<rows<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<"UNICODE_CHUNK_RUNNER_ERROR "<<e.what()<<"\n";return 1;}
}

#endif

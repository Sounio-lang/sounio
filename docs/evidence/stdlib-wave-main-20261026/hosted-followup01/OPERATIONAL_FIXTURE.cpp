#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
namespace f=std::filesystem;
void line(const char* key,const std::string& value) {
 const char* p=std::getenv(key);if(!p)return;
 int fd=open(p,O_WRONLY|O_CREAT|O_APPEND,0600);
 if(fd<0)std::exit(96);
 const auto text=value+"\n";if(write(fd,text.data(),text.size())!=static_cast<ssize_t>(text.size()))std::exit(97);close(fd);
}
int main(int argc,char**argv) {
 std::ifstream cfg(std::string(argv[0])+".mode");std::string kind;
 if(std::getline(cfg,kind)) {
  cfg.close();f::remove(std::string(argv[0])+".mode");
  line("STUB_NATIVE_LOG",kind+" "+argv[0]);
  if(kind=="native_timeout")std::this_thread::sleep_for(std::chrono::seconds(35));
  if(kind=="native_failure"){std::cout<<"PHASE_EXPECTED_FAILURE\n";return 7;}
  if(kind=="marker_missing"){std::cout<<"NATIVE_WITHOUT_EXPECTED_MARKER\n";return 0;}
  if(kind=="mutate") {
   f::remove(argv[0]);f::copy_file(std::getenv("STUB_COMPILER_BIN"),argv[0]);
   std::ofstream out(argv[0],std::ios::binary|std::ios::app);out.put('x');
  }
  std::cout<<"PHASE_NATIVE_OK\n";return 0;
 }
 if(argc==2&&std::string(argv[1])=="--version"){std::cout<<"Madaros v0.80.0 -- the Sounio self-hosted compiler\n";return 0;}
 if(argc!=4||std::string(argv[2])!="-o")return 93;
 kind=f::path(argv[1]).stem().string();
 if(kind.ends_with("_known"))kind.resize(kind.size()-6);
 if(kind.ends_with("_vacuous"))kind.resize(kind.size()-8);
 if(kind=="compile_xfail")kind="bad_rc";
 line("STUB_CALL_LOG",kind+" "+argv[3]);
 if(kind=="compile_timeout")std::this_thread::sleep_for(std::chrono::seconds(35));
 if(kind=="missing")return 0;
 if(kind=="badelf"){std::ofstream out(argv[3]);out<<"#!/bin/sh\necho BADSCRIPT_EXEC >> \"$STUB_NATIVE_LOG\"\n";return 0;}
 f::copy_file(argv[0],argv[3],f::copy_options::overwrite_existing);
 const auto runtime=(kind=="xfail"||kind=="xfail_wrong")?"native_failure":kind=="xfail_timeout"?"native_timeout":kind=="compiler_marker"?"marker_missing":kind;
 std::ofstream(std::string(argv[3])+".mode")<<runtime<<"\n";
 if(kind=="source_mutate"||kind=="source_mutate_bad_rc"){std::ofstream(argv[1],std::ios::app)<<"// changed during synthetic compile\n";}
 if(kind=="source_mutate_bad_rc")return 7;
 if(kind=="error0_no_colon"){std::cerr<<"error[E259] in synthetic metadata fixture\n";return 0;}
 if(kind=="error0"){std::cerr<<"error[E900]: synthetic compiler diagnostic\n";return 0;}
 if(kind=="bad_rc"){std::cerr<<"compiler deliberately returns7 after emitting ELF\n";return 7;}
 if(kind=="compiler_marker")std::cout<<"PHASE_NATIVE_OK\n";
 return 0;
}

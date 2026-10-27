#include <filesystem>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <string_view>
int main(int argc,char**argv) {
 if(argc==1){return 0;}
 const std::string_view first=argv[1];
 if(first=="--version"){std::cout<<"Madaros v0.80.0 test-double\n";return 0;}
 if(first=="--help"){std::cout<<"--check SOURCE --version\n";return 0;}
 if(first=="--check"){return 0;}
 if(argc!=4||std::string_view(argv[2])!="-o"){return 2;}
 const char* value=std::getenv("FAKE_COMPILER_MODE");
 const std::string_view mode=value?value:"success";
 if(mode!="empty-success"){std::filesystem::copy_file(argv[0],argv[3],std::filesystem::copy_options::overwrite_existing);}
 if(mode=="refusal"){std::cout<<"NATIVE_REFUSAL empty_stub_ud2 fn15 name=len missing_lowered_body\n";}
 if(mode!="no-success"){std::cout<<"Compilation successful!\n";}
 if(mode=="failure"){return 9;}
 if(mode=="signal"){std::raise(SIGILL);}
 return 0;
}

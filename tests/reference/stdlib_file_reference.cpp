#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <unistd.h>
using I=std::int64_t;
extern "C" I sounio_platform_call(I,I,I,I,I) noexcept;
I call(I op,I a=0,I b=0,I c=0,I d=0){return sounio_platform_call(op,a,b,c,d);}
I error(){return call(0);}
int main(){
 const char* path="native-file-reference.tmp";
 ::unlink(path);::close(STDIN_FILENO);
 I f=call(90,reinterpret_cast<I>(path),std::strlen(path),1);assert(error()==0&&f>0);
 unsigned char data[]={0,255,17,42,0};
 assert(call(92,f,reinterpret_cast<I>(data),5)==5&&error()==0);
 assert(call(94,f)==0&&error()==0);
 assert(call(93,f,0,0)==0&&error()==0);
 unsigned char read[8]{};
 assert(call(91,f,reinterpret_cast<I>(read),5)==0&&error()==9);
 assert(call(2,f)==0&&error()==0);
 assert(call(91,f,reinterpret_cast<I>(read),1)==0&&error()==9);
 f=call(90,reinterpret_cast<I>(path),std::strlen(path),0);assert(error()==0&&f>0);
 assert(call(91,f,reinterpret_cast<I>(read),8)==5&&error()==0);
 assert(std::memcmp(data,read,5)==0);
 assert(call(91,f,reinterpret_cast<I>(read),1)==0&&error()==0);
 assert(call(93,f,-1,0)==0&&error()==22);
 assert(call(93,f,0,7)==0&&error()==22);
 assert(call(93,f,1,0)==1&&error()==0);
 assert(call(91,f,reinterpret_cast<I>(read),1)==1&&error()==0&&read[0]==255);
 assert(call(2,f)==0&&error()==0);
 f=call(90,reinterpret_cast<I>(path),std::strlen(path),2);assert(error()==0);
 assert(call(92,f,reinterpret_cast<I>(data),2)==2&&error()==0);
 assert(call(2,f)==0&&error()==0);
 f=call(90,reinterpret_cast<I>(path),std::strlen(path),0);assert(error()==0);
 assert(call(91,f,reinterpret_cast<I>(read),8)==7&&error()==0);
 assert(read[5]==0&&read[6]==255);
 assert(call(2,f)==0&&error()==0);
 const char embedded[]={'a',0,'b'};
 assert(call(90,reinterpret_cast<I>(embedded),3,0)==0&&error()==22);
 assert(call(90,reinterpret_cast<I>(path),std::strlen(path),9)==0&&error()==22);
 assert(call(90,0,1,0)==0&&error()==22);
 assert(call(3)==0);
 char text[]={'a','b','c'};
 I owner=call(95,reinterpret_cast<I>(text),3,4);assert(error()==0&&owner>0);
 text[0]='z';const auto* copy=reinterpret_cast<const char*>(call(96,owner));
 assert(error()==0&&std::strcmp(copy,"abc")==0);
 assert(call(95,reinterpret_cast<I>(text),3,3)==0&&error()==105);
 const char zero[]={0};assert(call(95,reinterpret_cast<I>(zero),1,2)==0&&error()==84);
 assert(call(2,owner)==0&&error()==0);
 assert(call(96,owner)==0&&error()==9);
 assert(call(3)==0);
 ::unlink(path);std::puts("NATIVE_FILE_CPP_OK");
}

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
using I=std::int64_t;
extern "C" I sounio_platform_call(I,I,I,I,I) noexcept;
I call(I op,I a=0,I b=0,I c=0,I d=0){return sounio_platform_call(op,a,b,c,d);}
I error(){return call(0);}
int main(){
 I listener=call(50,2130706433,0,4);assert(error()==0&&listener>0);
 I port=call(51,listener);assert(error()==0&&port>0);
 assert(call(57,listener,1)==0&&error()==0);
 assert(call(57,listener,0)==0&&error()==22);
 I client=call(59,2130706433,port);assert(error()==0&&client>0);
 assert(call(58,client)==1&&error()==0);
 const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 I status=11;
 while(status==11&&std::chrono::steady_clock::now()<deadline){call(64,client);status=error();std::this_thread::yield();}
 assert(status==0);
 assert(call(64,client)==0&&error()==0);
 I ready=0;
 do{ready=call(57,listener,1);assert(error()==0);std::this_thread::yield();}while(!(ready&1)&&std::chrono::steady_clock::now()<deadline);
 assert(ready&1);
 I server=call(53,listener,0);assert(error()==0&&server>0);
 assert(call(57,server,1)==0&&error()==0);
 unsigned char value=42,received=0;
 assert(call(54,client,reinterpret_cast<I>(&value),1,0)==1&&error()==0);
 do{ready=call(57,server,1);assert(error()==0);std::this_thread::yield();}while(!(ready&1)&&std::chrono::steady_clock::now()<deadline);
 assert(ready&1);
 assert(call(55,server,reinterpret_cast<I>(&received),1,0)==1&&error()==0&&received==42);
 assert(call(2,server)==0&&error()==0);
 assert(call(57,server,1)==0&&error()==9);
 assert(call(2,client)==0&&error()==0);assert(call(2,listener)==0&&error()==0);
 I failed=call(59,2130706433,0);I failure=error();
 if(failed>0){
  failure=11;
  while(failure==11&&std::chrono::steady_clock::now()<deadline){call(64,failed);failure=error();std::this_thread::yield();}
  assert(failure==111);
  call(64,failed);assert(error()==111);
  assert(call(2,failed)==0&&error()==0);
 }else assert(failure==111);
 I udp=call(60,2130706433,0);assert(error()==0);
 assert(call(58,udp)==2&&error()==0);
 assert(call(64,udp)==0&&error()==107);
 assert(call(2,udp)==0&&error()==0);
 assert(call(3)==0);std::puts("SOCKET_READY_CPP_OK");
}

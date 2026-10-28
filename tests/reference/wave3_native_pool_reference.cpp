#include <dirent.h>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <thread>
using I=std::int64_t;
extern "C" I sounio_platform_call(I,I,I,I,I) noexcept;
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
static std::atomic<int> fail_after{-1},created{0},entered{0},active{0};
static std::atomic<bool> release_workers{false};
extern "C" int __wrap_pthread_create(pthread_t* t,const pthread_attr_t* a,void*(*f)(void*),void* p) {
  const int attempt=created.fetch_add(1);
  if(fail_after.load()>=0&&attempt>=fail_after.load())return EAGAIN;
  return __real_pthread_create(t,a,f,p);
}
static I square(I x){return x*x;}
static I held(I x){
  ++active;++entered;
  while(!release_workers.load())std::this_thread::yield();
  --active;return x*x;
}
struct R{I value,error;};
static R call(I op,I a=0,I b=0,I c=0,I d=0){
  const I value=sounio_platform_call(op,a,b,c,d);
  const I error=sounio_platform_call(0,0,0,0,0);return {value,error};
}
static I ok(I op,I a=0,I b=0,I c=0,I d=0){auto r=call(op,a,b,c,d);assert(r.error==0);return r.value;}
static void err(I expected,I op,I a=0,I b=0,I c=0,I d=0){assert(call(op,a,b,c,d).error==expected);}
static void release_pool(I pool){ok(133,pool,0);ok(134,pool);ok(136,pool);ok(2,pool);}

struct Pair {I listener,client,server;};
static Pair tcp_pair(){
 const I listener=ok(50,2130706433,0,4),port=ok(51,listener);
 const I client=ok(52,2130706433,port,1000),server=ok(53,listener,1000);
 return {listener,client,server};
}
static void close_pair(Pair p){if(p.server)ok(2,p.server);ok(2,p.client);ok(2,p.listener);}
static int fd_count(){
 DIR* directory=opendir("/proc/self/fd");assert(directory);int count=0;
 while(readdir(directory))++count;
 closedir(directory);return count;
}
static void tcp_readers(){
 const int before=fd_count();
 char out[8]{};
 auto first=tcp_pair();
 const I reader=ok(138,first.server,ok(30)+1000000000,3);
 err(EBUSY,138,first.server,ok(30)+1000000000,3);
 err(EBUSY,55,first.server,reinterpret_cast<I>(out),1,0);
 err(EBUSY,111,first.server,reinterpret_cast<I>(out),1,0);
 err(EBUSY,56,first.server,0);
 err(EBADF,92,reader,reinterpret_cast<I>(out),1);
 err(ESPIPE,93,reader,0,0);
 const char abc[]="abc";
 assert(ok(54,first.client,reinterpret_cast<I>(abc),3,1000)==3);
 ok(56,first.client,1);
 ok(2,first.server);first.server=0;
 assert(ok(91,reader,reinterpret_cast<I>(out),8)==3);
 assert(out[0]=='a'&&out[1]=='b'&&out[2]=='c');
 assert(ok(91,reader,reinterpret_cast<I>(out),8)==0);
 ok(2,reader);close_pair(first);
 auto second=tcp_pair();
 const I limited=ok(138,second.server,ok(30)+1000000000,1);
 const char ab[]="ab";
 assert(ok(54,second.client,reinterpret_cast<I>(ab),2,1000)==2);
 ok(56,second.client,1);
 assert(ok(91,limited,reinterpret_cast<I>(out),8)==1&&out[0]=='a');
 err(ENOBUFS,91,limited,reinterpret_cast<I>(out),8);
 assert(ok(91,limited,reinterpret_cast<I>(out),0)==0);
 ok(2,limited);
 assert(ok(55,second.server,reinterpret_cast<I>(out),8,1000)==1&&out[0]=='b');
 assert(ok(55,second.server,reinterpret_cast<I>(out),8,1000)==0);
 close_pair(second);
 auto third=tcp_pair();
 const I deadline=ok(30)+2000000;
 const I timed=ok(138,third.server,deadline,8);
 err(ETIMEDOUT,91,timed,reinterpret_cast<I>(out),8);
 assert(ok(54,third.client,reinterpret_cast<I>(abc),1,1000)==1);
 err(ETIMEDOUT,91,timed,reinterpret_cast<I>(out),8);
 ok(2,timed);
 assert(ok(55,third.server,reinterpret_cast<I>(out),8,1000)==1&&out[0]=='a');
 err(EINVAL,138,third.server,-1,8);
 err(EINVAL,138,third.server,0,-1);
 err(ENOTCONN,138,third.listener,0,8);
 close_pair(third);
 const I udp=ok(60,2130706433,0);
 err(EPROTOTYPE,138,udp,0,8);ok(2,udp);
 assert(ok(3)==0);
 assert(fd_count()==before);
 std::puts("TCP_READER_CPP_OK lease=exclusive dup=independent deadline=absolute excess=peeked");
}

int main(){
  const I reserved=ok(139,2,4);
  err(EINVAL,139,0,4);err(EINVAL,139,65,4);err(EINVAL,139,2,0);
  err(ENOBUFS,130,reinterpret_cast<I>(square),2,4,reserved-1);
  err(EINVAL,130,0,2,4,reserved);
  err(EINVAL,130,1,2,4,reserved);
  created=0;fail_after=1;
  err(EAGAIN,130,reinterpret_cast<I>(square),2,4,reserved);
  assert(ok(3)==0&&active==0);
  fail_after=-1;created=0;
  const I pool=ok(130,reinterpret_cast<I>(held),2,4,reserved);
  assert(ok(135,pool,1)==2&&ok(135,pool,2)==4);
  err(EBUSY,2,pool);
  for(I i=0;i<4;++i)assert(ok(131,pool,i,0)==i);
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(entered<2&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
  assert(entered==2&&ok(135,pool,4)==2&&ok(135,pool,0)==4);
  err(EAGAIN,131,pool,99,0);err(ETIMEDOUT,131,pool,99,2);
  err(EAGAIN,132,pool,0,0);err(ETIMEDOUT,132,pool,0,2);
  err(EINVAL,131,pool,0,-1);err(ENOENT,132,pool,1234,0);
  assert(ok(133,pool,1)==2);
  err(EPIPE,131,pool,100,0);
  release_workers=true;ok(134,pool);
  assert(ok(132,pool,0,0)==0&&ok(132,pool,1,0)==1);
  err(ECANCELED,132,pool,2,0);err(ECANCELED,132,pool,3,0);
  err(ENOENT,132,pool,0,0);err(EINVAL,134,pool);
  assert(ok(135,pool,0)==0);ok(2,pool);assert(ok(3)==0);
  const I p2=ok(130,reinterpret_cast<I>(square),4,17,ok(139,4,17));
  I expected_sum=0,actual_sum=0;
  for(I base=0;base<10000;base+=16){
    const I count=10000-base<16?10000-base:16;
    for(I j=0;j<count;++j){const I x=base+j-5000;assert(ok(131,p2,x,1000)==base+j);expected_sum+=x*x;}
    for(I j=count;j>0;--j){const I x=base+j-1-5000;const I v=ok(132,p2,base+j-1,1000);assert(v==x*x);actual_sum+=v;}
  }
  assert(actual_sum==expected_sum);
  release_pool(p2);assert(ok(3)==0);
  const I p3=ok(130,reinterpret_cast<I>(square),1,1,ok(139,1,1));
  ok(131,p3,7,0);ok(133,p3,0);ok(134,p3);
  assert(ok(135,p3,0)==1);err(EBUSY,2,p3);assert(ok(136,p3)==1);ok(2,p3);
  assert(ok(3)==0);
  tcp_readers();
  std::puts("NATIVE_POOL_CPP_OK tasks=10000 peak=2 cleanup=0 partial_spawn=joined");
}

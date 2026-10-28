// Independent C++23 protocol controls. No Sounio callback runs in this worker.
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <dirent.h>
#include <limits>
#include <pthread.h>
#include <thread>
using I=std::int64_t;
extern "C" I sounio_platform_call(I,I,I,I,I) noexcept;
extern "C" int __real_pthread_create(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
static std::atomic<int> fail_after{-1},created{0};
extern "C" int __wrap_pthread_create(pthread_t* t,const pthread_attr_t* a,void*(*f)(void*),void* p){
 const int attempt=created.fetch_add(1);
 if(fail_after.load()>=0&&attempt>=fail_after.load())return EAGAIN;
 return __real_pthread_create(t,a,f,p);
}
struct R{I value,error;};
static R call(I op,I a=0,I b=0,I c=0,I d=0){
 const I v=sounio_platform_call(op,a,b,c,d),e=sounio_platform_call(0,0,0,0,0);return{v,e};
}
static I ok(I op,I a=0,I b=0,I c=0,I d=0){const auto r=call(op,a,b,c,d);assert(r.error==0);return r.value;}
static void err(I expected,I op,I a=0,I b=0,I c=0,I d=0){assert(call(op,a,b,c,d).error==expected);}
static int entries(const char* path){DIR* d=opendir(path);assert(d);int n=0;while(readdir(d))++n;closedir(d);return n;}
static I channel(I cap=2){return ok(70,cap,cap*8);}
static I builder(I ch){return ok(141,ch,ok(149));}
static I producer(I ch,I first,I count,I step,I deadline=0){
 const I p=builder(ch);ok(142,p,first,count,step);
 ok(143,p,deadline?deadline:ok(30)+5000000000);return p;
}
static void joined(I p,I count,I work_error=0){
 assert(ok(144,p)==count);assert(ok(145,p)==work_error);
 assert(ok(147,p,0)==count&&ok(147,p,5)==1&&ok(147,p,6)==1);
 err(EINVAL,144,p);ok(2,p);
}
template<std::size_t N>static void sequence(I first,I step,const std::array<I,N>& expected){
 const I c=channel(),p=producer(c,first,static_cast<I>(N),step);
 for(const I v:expected)assert(ok(74,c,1000)==v);
 joined(p,static_cast<I>(N));err(EAGAIN,72,c);ok(2,c);assert(ok(3)==0);
}
static void invalid_and_budget(){
 const I c=channel(),bytes=ok(149);
 assert(bytes>1048576);err(EINVAL,149,1);
 err(EINVAL,141,c,-1);err(ENOBUFS,141,c,bytes-1);
 err(EBADF,141,-1,bytes);err(EINVAL,141,c,bytes,1);
 I p=builder(c);err(EINVAL,144,p);err(EAGAIN,145,p);
 err(EINVAL,142,p,0,-1,1);
 err(EOVERFLOW,142,p,std::numeric_limits<I>::max(),2,1);
 err(EOVERFLOW,142,p,std::numeric_limits<I>::min(),2,-1);
 err(EOVERFLOW,142,p,std::numeric_limits<I>::max(),3,std::numeric_limits<I>::min());
 ok(142,p,0,0,0);err(EBUSY,142,p,0,0,0);
 err(EINVAL,143,p,0);err(ETIMEDOUT,143,p,ok(30)-1);
 ok(2,p);err(EBADF,143,p,ok(30)+1000000);
 assert(ok(3)==1);ok(2,c);assert(ok(3)==0);
 std::printf("CHANNEL_RANGE_BUDGET bytes=%lld\n",static_cast<long long>(bytes));
}
static void basic_and_backpressure(){
 const I c=channel(),p=producer(c,0,128,1);
 err(EBUSY,143,p,ok(30)+1000000000);err(EBUSY,2,p);
 for(I i=0;i<128;++i)assert(ok(74,c,1000)==i);
 joined(p,128);ok(2,c);assert(ok(3)==0);
 sequence(-7,0,std::array<I,3>{-7,-7,-7});
 sequence(3,-1,std::array<I,4>{3,2,1,0});
 sequence(-4,2,std::array<I,4>{-4,-2,0,2});
 sequence(std::numeric_limits<I>::max(),std::numeric_limits<I>::min(),
          std::array<I,2>{std::numeric_limits<I>::max(),-1});
 const I zero_channel=channel(),zero=producer(zero_channel,std::numeric_limits<I>::min(),0,
                                             std::numeric_limits<I>::min());
 joined(zero,0);err(EAGAIN,72,zero_channel);ok(2,zero_channel);
 assert(ok(3)==0);
}
static void deadline_and_close(){
 I c=channel(1);ok(71,c,99);
 I p=producer(c,0,8,1,ok(30)+20000000);
 joined(p,0,ETIMEDOUT);assert(ok(72,c)==99);err(EAGAIN,72,c);ok(2,c);
 c=channel(1);ok(71,c,99);p=producer(c,0,8,1);
 ok(75,c);joined(p,0,EPIPE);assert(ok(72,c)==99);err(ENODATA,72,c);ok(2,c);
 c=channel(1);ok(71,c,99);p=producer(c,0,8,1);
 ok(2,c);joined(p,0,EBADF);assert(ok(3)==0);
 // A successful prefix does not restart the original absolute deadline.
 c=channel(1);ok(71,c,99);const I deadline=ok(30)+1000000000;
 p=producer(c,0,8,1,deadline);
 std::this_thread::sleep_for(std::chrono::milliseconds(600));
 assert(ok(72,c)==99);
 const auto progress_end=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
 while(ok(147,p,0)==0&&std::chrono::steady_clock::now()<progress_end)std::this_thread::yield();
 assert(ok(147,p,0)==1);
 while(ok(30)<deadline+300000000)std::this_thread::sleep_for(std::chrono::milliseconds(1));
 assert(ok(147,p,6)==1);
 joined(p,1,ETIMEDOUT);assert(ok(72,c)==0);err(EAGAIN,72,c);ok(2,c);
 assert(ok(3)==0);
}
static void cancellation_and_concurrent_join(){
 I c=channel(1);ok(71,c,99);
 I p=producer(c,std::numeric_limits<I>::min(),std::numeric_limits<I>::max(),0);
 ok(146,p);const I at_cancel=ok(147,p,0);
 assert(at_cancel==0);joined(p,at_cancel,ECANCELED);
 assert(ok(72,c)==99);err(EAGAIN,72,c);ok(2,c);
 c=channel(1);ok(71,c,99);p=producer(c,7,10,0);
 std::array<R,2> results{};
 std::thread a([&]{results[0]=call(144,p);});
 std::thread b([&]{results[1]=call(144,p);});
 const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
 while(ok(147,p,6)==0&&std::chrono::steady_clock::now()<until){
  // At least one join is blocked on this full channel; cancellation releases it.
  std::this_thread::yield();break;
 }
 ok(146,p);a.join();b.join();
 const bool first=results[0].error==0&&results[0].value==0&&results[1].error==EINVAL;
 const bool second=results[1].error==0&&results[1].value==0&&results[0].error==EINVAL;
 assert(first||second);assert(ok(145,p)==ECANCELED);
 ok(2,p);assert(ok(72,c)==99);ok(2,c);assert(ok(3)==0);
}
static void maximum_deadline(){
 const I c=channel(1);ok(71,c,99);
 const I p=producer(c,0,1,1,std::numeric_limits<I>::max());
 const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
 while(ok(147,p,9)==0&&std::chrono::steady_clock::now()<until)std::this_thread::yield();
 assert(ok(147,p,9)>0);
 ok(146,p);joined(p,0,ECANCELED);
 assert(ok(72,c)==99);ok(2,c);assert(ok(3)==0);
}
static void partial_construction(){
 const I c=channel(1);ok(71,c,99);
 const I live=producer(c,0,2,1);
 const I pending=builder(c);ok(142,pending,0,3,1);
 created=0;fail_after=0;
 err(EAGAIN,143,pending,ok(30)+1000000000);
 assert(created==1&&ok(147,pending,4)==0);
 fail_after=-1;ok(2,pending);assert(ok(3)==2);
 ok(146,live);joined(live,0,ECANCELED);
 assert(ok(72,c)==99);ok(2,c);assert(ok(3)==0);
}
static void two_leases(){
 const I c=channel(),a=producer(c,0,64,1),b=producer(c,1000,64,1);
 I next_a=0,next_b=1000;
 for(I i=0;i<128;++i){
  const I x=ok(74,c,1000);
  if(x<1000){assert(x==next_a);++next_a;}
  else{assert(x==next_b);++next_b;}
 }
 assert(next_a==64&&next_b==1064);joined(a,64);joined(b,64);ok(2,c);
 assert(ok(3)==0);
}
int main(){
 const int fds=entries("/proc/self/fd"),threads=entries("/proc/self/task");
 invalid_and_budget();basic_and_backpressure();deadline_and_close();
 cancellation_and_concurrent_join();maximum_deadline();partial_construction();two_leases();
 const auto cleanup_end=std::chrono::steady_clock::now()+std::chrono::seconds(1);
 while(entries("/proc/self/task")!=threads&&std::chrono::steady_clock::now()<cleanup_end)std::this_thread::sleep_for(std::chrono::milliseconds(1));
 assert(entries("/proc/self/fd")==fds&&entries("/proc/self/task")==threads&&ok(3)==0);
 std::puts("NATIVE_CHANNEL_RANGE_CPP_OK zero=exact range=general lease=retained deadline=absolute cancel=linearized partial_spawn=clean cleanup=0");
}

#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <iostream>
#include <thread>
extern "C" std::int64_t sounio_platform_call(std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t) noexcept;
using I=std::int64_t;
static I call(I op,I a=0,I b=0,I c=0,I d=0){return sounio_platform_call(op,a,b,c,d);}
static I error(){return call(0);}
static std::array<std::atomic<int>,40000> seen{};
static std::atomic<I> ticket{0};
static I produce(I channel) {
 for(int i=0;i<10000;i++){const I v=ticket.fetch_add(1);call(73,channel,v,5000);assert(error()==0);}
 return 10000;
}
static I consume(I channel) {
 for(int i=0;i<20000;i++){const I v=call(74,channel,5000);assert(error()==0);assert(v>=0&&v<40000);assert(seen[static_cast<std::size_t>(v)].fetch_add(1)==0);}
 return 20000;
}
int main(){
 assert(call(80)==1&&error()==0);
 call(70,0,0);assert(error()==EINVAL);
 call(70,2,8);assert(error()==ENOBUFS);
 I q=call(70,2,16);assert(q>0&&error()==0);
 call(81);assert(error()==EBUSY);
 assert(call(80)==2&&error()==0);call(81);assert(error()==0);
 call(72,q);assert(error()==EAGAIN);
 call(74,q,1);assert(error()==ETIMEDOUT);
 call(74,q,-1);assert(error()==EINVAL);
 call(71,q,0);assert(error()==0);call(71,q,-7);assert(error()==0);
 call(71,q,8);assert(error()==EAGAIN);
 call(73,q,8,1);assert(error()==ETIMEDOUT);
 assert(call(72,q)==0&&error()==0);
 assert(call(72,q)==-7&&error()==0);
 call(71,q,42);assert(error()==0);
 call(75,q);assert(error()==0);
 call(71,q,1);assert(error()==EPIPE);
 assert(call(72,q)==42&&error()==0);
 call(72,q);assert(error()==ENODATA);
 call(75,q);assert(error()==0);
 call(2,q);assert(error()==0);call(76,q);assert(error()==EBADF);
 q=call(70,64,512);assert(q>0&&error()==0);
 std::array<I,6> threads{};
 for(std::size_t i=0;i<4;i++){threads[i]=call(20,reinterpret_cast<I>(&produce),q);assert(threads[i]>0&&error()==0);}
 for(std::size_t i=4;i<6;i++){threads[i]=call(20,reinterpret_cast<I>(&consume),q);assert(threads[i]>0&&error()==0);}
 for(std::size_t i=0;i<6;i++){const I expected=i<4?10000:20000;assert(call(21,threads[i])==expected&&error()==0);call(2,threads[i]);assert(error()==0);}
 assert(ticket.load()==40000);for(const auto& n:seen)assert(n.load()==1);
 assert(call(77,q)==0&&error()==0);
 call(2,q);assert(error()==0);
 q=call(70,1,8);assert(q>0&&error()==0);
 std::atomic<bool> entered{false};
 std::thread waiter([&]{entered.store(true);call(74,q,5000);assert(error()==EBADF);});
 while(!entered.load())std::this_thread::yield();
 call(2,q);assert(error()==0);waiter.join();
 assert(call(3)==0&&error()==0);call(81);assert(error()==0);call(81);assert(error()==EINVAL);
 std::cout<<"NATIVE_CHANNEL_CPP_OK\n";
}

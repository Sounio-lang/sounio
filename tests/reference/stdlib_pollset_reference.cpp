#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <pthread.h>
extern "C" std::int64_t sounio_platform_call(std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t) noexcept;
using I=std::int64_t;
static I call(I op,I a=0,I b=0,I c=0,I d=0){return sounio_platform_call(op,a,b,c,d);}
static I error(){return call(0);}
static I now(){const I t=call(30);assert(t>0&&error()==0);return t;}
static void signal_handler(int){}
int main(){
 const I ip1=2130706433,ip2=2130706434;
 assert(call(100,2,47)==0&&error()==ENOBUFS);
 assert(call(100,0,100)==0&&error()==EINVAL);
 assert(call(100,std::numeric_limits<I>::max(),std::numeric_limits<I>::max())==0&&error()==EOVERFLOW);
 const I receiver=call(60,ip2,0),sender=call(60,ip1,0);
 assert(receiver>0&&sender>receiver&&error()==0);
 const I port=call(51,receiver),sender_port=call(51,sender);
 assert(port>0&&sender_port>0);
 const I set=call(100,2,48);assert(set>0&&error()==0);
 assert(call(101,set,receiver,1,91)==1&&error()==0);
 assert(call(103,set,0)==0&&error()==0);
 assert(call(104,set,0)==0&&error()==ERANGE);
 assert(call(101,set,sender,2,92)==1);
 const I extra=call(60,ip1,0);assert(extra>0);
 assert(call(101,set,extra,1,93)==0&&error()==ENOBUFS);
 assert(call(101,set,extra,0,93)==0&&error()==EINVAL);
 const char payload[]="abc";
 assert(call(108,sender,reinterpret_cast<I>(payload),3)==0&&error()==EDESTADDRREQ);
 assert(call(107,sender,4294967296LL,port)==0&&error()==EINVAL);
 assert(call(107,sender,ip2,port)==1&&error()==0);
 assert(call(108,sender,reinterpret_cast<I>(payload),3)==3&&error()==0);
 assert(call(103,set,now()+1000000000)==2&&error()==0);
 assert(call(104,set,0)==receiver&&call(105,set,0)==1&&call(106,set,0)==91);
 assert(call(104,set,1)==sender&&(call(105,set,1)&2)&&call(106,set,1)==92);
 char buffer[16]{};
 assert(call(62,receiver,reinterpret_cast<I>(buffer),sizeof(buffer),1000)==3&&error()==0);
 assert(std::memcmp(buffer,payload,3)==0);
 assert(call(109,receiver)==ip1&&call(63,receiver)==sender_port);
 assert(call(108,sender,0,0)==0&&error()==0);
 assert(call(62,receiver,0,0,1000)==0&&error()==0);
 assert(call(109,receiver)==ip1);
 assert(call(108,sender,reinterpret_cast<I>(payload),3)==3);
 assert(call(62,receiver,reinterpret_cast<I>(buffer),1,1000)==1&&error()==EMSGSIZE);
 assert(call(109,receiver)==ip1&&call(63,receiver)==sender_port);
 assert(call(102,set,sender)==1&&error()==0);
 assert(call(102,set,sender)==0&&error()==ENOENT);
 assert(call(101,set,receiver,1,191)==1);
 // EINTR does not restart the absolute deadline.
 struct sigaction action{};action.sa_handler=signal_handler;sigemptyset(&action.sa_mask);
 assert(sigaction(SIGUSR1,&action,nullptr)==0);
 const pthread_t waiting=pthread_self();
 std::thread signals([waiting]{for(int i=0;i<8;++i){std::this_thread::sleep_for(std::chrono::milliseconds(3));pthread_kill(waiting,SIGUSR1);}});
 const I started=now();
 assert(call(103,set,started+30000000)==0&&error()==0);
 const I elapsed=now()-started;
 signals.join();assert(elapsed>=20000000&&elapsed<300000000);
 // Closing the logical registration during a wait cannot expose a reused fd.
 std::thread closer([receiver]{std::this_thread::sleep_for(std::chrono::milliseconds(5));call(2,receiver);assert(error()==0);});
 assert(call(103,set,now()+30000000)==1&&error()==0);
 closer.join();
 assert(call(104,set,0)==receiver&&call(105,set,0)==16&&call(106,set,0)==191);
 const I replacement=call(60,ip1,0);assert(replacement>receiver);
 assert(call(103,set,0)==1&&call(105,set,0)==16);
 assert(call(102,set,receiver)==1);
 assert(call(103,set,0)==0&&error()==0);
 call(2,set);assert(error()==0);
 assert(call(103,set,0)==0&&error()==EBADF);
 const I listener=call(50,ip1,0,8);assert(listener>0);
 const I tcp_port=call(51,listener);
 const I client=call(52,ip1,tcp_port,1000);assert(client>0);
 const I accepted=call(53,listener,1000);assert(accepted>0);
 assert(call(107,client,ip1,port)==0&&error()==EPROTOTYPE);
 assert(call(110,receiver,0,0,0)==0&&error()==EBADF);
 assert(call(110,sender,0,0,0)==0&&error()==EPROTOTYPE);
 assert(call(110,client,0,1,now())==0&&error()==EINVAL);
 assert(call(111,accepted,reinterpret_cast<I>(buffer),1,-1)==0&&error()==EINVAL);
 assert(call(110,client,0,0,0)==0&&error()==0);
 assert(call(110,client,reinterpret_cast<I>(payload),3,now()+1000000000)==3&&error()==0);
 assert(call(111,accepted,reinterpret_cast<I>(buffer),sizeof(buffer),now()+1000000000)==3&&error()==0);
 assert(std::memcmp(buffer,payload,3)==0);
 assert(call(111,accepted,reinterpret_cast<I>(buffer),1,now()+10000000)==0&&error()==ETIMEDOUT);
 assert(call(56,client,1)==0&&error()==0);
 assert(call(111,accepted,reinterpret_cast<I>(buffer),1,now()+1000000000)==0&&error()==0);
 for(I id:{sender,extra,replacement,listener,client,accepted}){call(2,id);assert(error()==0);}
 std::cout<<"STDLIB_POLLSET_UDP_ABSOLUTE_DEADLINE_OK\n";
}

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
extern "C" std::int64_t sounio_platform_call(std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t) noexcept;
using I=std::int64_t;
static I call(I op,I a=0,I b=0,I c=0,I d=0){return sounio_platform_call(op,a,b,c,d);}
static I error(){return call(0);}
static I increment(I atomic) {
    for(int n=0;n<10000;n++){call(13,atomic,1);assert(error()==0);}
    return -7;
}
static I guarded_count=0;
static I increment_mutex(I mutex) {
    for(int n=0;n<10000;n++){call(17,mutex);assert(error()==0);guarded_count++;call(19,mutex);assert(error()==0);}
    return 0;
}
int main() {
    assert(call(1)==1&&error()==0);
    const I atomic=call(10,0);assert(atomic>0&&error()==0);
    assert(call(11,atomic)==0&&error()==0);
    I threads[4]{};
    for(auto& t:threads){t=call(20,reinterpret_cast<I>(&increment),atomic);assert(t>0&&error()==0);}
    call(2,threads[0]);assert(error()==EBUSY);
    for(auto t:threads){assert(call(21,t)==-7&&error()==0);call(21,t);assert(error()==EINVAL);call(2,t);assert(error()==0);}
    assert(call(11,atomic)==40000&&error()==0);
    assert(call(14,atomic,40000,0)==1&&error()==0);
    assert(call(14,atomic,40000,7)==0&&error()==0);
    assert(call(11,atomic)==0&&error()==0);
    assert(call(12,atomic,-123)==0&&error()==0);
    assert(call(13,atomic,3)==-123&&error()==0);
    assert(call(11,atomic)==-120&&error()==0);
    call(2,atomic);assert(error()==0);call(11,atomic);assert(error()==EBADF);
    const I mutex=call(16);assert(mutex>0&&error()==0);
    call(17,mutex);assert(error()==0);
    call(17,mutex);assert(error()==EDEADLK);
    call(18,mutex);assert(error()==EBUSY);
    call(2,mutex);assert(error()==EBUSY);
    call(19,mutex);assert(error()==0);
    call(19,mutex);assert(error()==EPERM);
    for(auto& t:threads){t=call(20,reinterpret_cast<I>(&increment_mutex),mutex);assert(t>0&&error()==0);}
    for(auto t:threads){assert(call(21,t)==0&&error()==0);call(2,t);assert(error()==0);}
    assert(guarded_count==40000);
    call(2,mutex);assert(error()==0);call(17,mutex);assert(error()==EBADF);
    const I start=call(30);assert(start>0&&error()==0);assert(call(30)>=start&&error()==0);
    unsigned char bytes[64]{};
    assert(call(40,reinterpret_cast<I>(bytes),64)==64&&error()==0);
    assert(call(40,0,0)==0&&error()==0);
    call(40,0,1);assert(error()==EINVAL);
    const I listener=call(50,2130706433,0,8);assert(listener>0&&error()==0);
    const I port=call(51,listener);assert(port>0&&error()==0);
    const I client=call(52,2130706433,port,1000);assert(client>0&&error()==0);
    const I server=call(53,listener,1000);assert(server>0&&error()==0);
    unsigned char message[]{'h','e','l','l','o'}, out[8]{};
    call(55,server,reinterpret_cast<I>(out),8,2);assert(error()==ETIMEDOUT);
    assert(call(54,client,reinterpret_cast<I>(message),5,1000)==5&&error()==0);
    assert(call(55,server,reinterpret_cast<I>(out),3,1000)==3&&error()==0);
    assert(std::memcmp(out,message,3)==0);
    assert(call(55,server,reinterpret_cast<I>(out),8,1000)==2&&error()==0);
    assert(std::memcmp(out,message+3,2)==0);
    call(56,client,1);assert(error()==0);
    assert(call(55,server,reinterpret_cast<I>(out),8,1000)==0&&error()==0);
    for(const auto id:{server,client,listener}){call(2,id);assert(error()==0);}
    const I receiver=call(60,2130706433,0),sender=call(60,2130706433,0);assert(receiver>0&&sender>0&&error()==0);
    const I dest=call(51,receiver),source=call(51,sender);
    assert(call(61,sender,reinterpret_cast<I>(message),5,dest)==5&&error()==0);
    assert(call(62,receiver,reinterpret_cast<I>(out),8,1000)==5&&error()==0);
    assert(std::memcmp(out,message,5)==0);
    assert(call(63,receiver)==source&&error()==0);
    assert(call(61,sender,reinterpret_cast<I>(message),5,dest)==5&&error()==0);
    assert(call(62,receiver,reinterpret_cast<I>(out),3,1000)==3&&error()==EMSGSIZE);
    assert(std::memcmp(out,message,3)==0);
    call(62,receiver,reinterpret_cast<I>(out),8,2);assert(error()==ETIMEDOUT);
    assert(call(61,sender,0,0,dest)==0&&error()==0);
    assert(call(62,receiver,0,0,1000)==0&&error()==0);
    call(2,sender);assert(error()==0);call(2,receiver);assert(error()==0);
    call(999);assert(error()==ENOSYS);assert(call(3)==0&&error()==0);
    std::cout<<"STDLIB_PLATFORM_CPP_OK\n";
}

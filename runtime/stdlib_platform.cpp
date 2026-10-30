// Minimal Linux platform ABI for the Sounio stdlib. Algorithms stay in Sounio.
// All entrypoints are noexcept. Operation 0 reads this thread's last errno;
// every other operation resets it. Resource IDs are never raw addresses.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>
#include <unistd.h>

#if !defined(__linux__) || !defined(__x86_64__)
#error "The native Sounio callback adapter currently targets Linux x86-64."
#endif
// Sounio's internal convention may use RBX/R12-R15 as scratch. A foreign C++
// caller must retain all SysV callee-saved registers across that callback.
// Six saved words plus one alignment word keep RSP 16-byte aligned at CALL.
extern "C" std::int64_t sounio_native_callback_thunk(std::int64_t(*)(std::int64_t),std::int64_t);
asm(
".text\n"
".p2align 4\n"
".globl sounio_native_callback_thunk\n"
".hidden sounio_native_callback_thunk\n"
".type sounio_native_callback_thunk,@function\n"
"sounio_native_callback_thunk:\n"
"pushq %rbp\n"
"pushq %rbx\n"
"pushq %r12\n"
"pushq %r13\n"
"pushq %r14\n"
"pushq %r15\n"
"subq $8,%rsp\n"
"movq %rdi,%rax\n"
"movq %rsi,%rdi\n"
"call *%rax\n"
"addq $8,%rsp\n"
"popq %r15\n"
"popq %r14\n"
"popq %r13\n"
"popq %r12\n"
"popq %rbx\n"
"popq %rbp\n"
"ret\n"
".size sounio_native_callback_thunk,.-sounio_native_callback_thunk\n"
);

namespace {
using I=std::int64_t;
thread_local int last_error=0;
enum class Kind {atomic,thread,socket,mutex};
struct Resource {const Kind kind;explicit Resource(Kind k):kind(k){};virtual ~Resource()=default;};
struct Atomic final:Resource {std::atomic<I> value;explicit Atomic(I v):Resource(Kind::atomic),value(v){}};
struct Thread final:Resource {
  using Callback=I(*)(I);
  pthread_t id{};Callback callback;I argument,result=0;
  std::mutex mutex;bool started=false,joining=false,joined=false;
  Thread(Callback f,I a):Resource(Kind::thread),callback(f),argument(a){}
  ~Thread() override {if(started&&!joined)pthread_detach(id);}
};
struct Mutex final:Resource {
  pthread_mutex_t native{};std::atomic<I> users{0};bool initialized=false;
  Mutex():Resource(Kind::mutex){}
  int initialize() {
    pthread_mutexattr_t attr;
    int rc=pthread_mutexattr_init(&attr);if(rc)return rc;
    rc=pthread_mutexattr_settype(&attr,PTHREAD_MUTEX_ERRORCHECK);
    if(rc==0)rc=pthread_mutex_init(&native,&attr);
    pthread_mutexattr_destroy(&attr);initialized=rc==0;return rc;
  }
  ~Mutex()override {if(initialized)pthread_mutex_destroy(&native);}
};
struct Socket final:Resource {
  int fd;std::atomic<int> peer_port{0};
  explicit Socket(int f):Resource(Kind::socket),fd(f){}
  ~Socket()override {if(fd>=0)::close(fd);}
};
std::mutex registry_mutex;
std::unordered_map<I,std::shared_ptr<Resource>> registry;
I next_id=1;
I failure(int code) {last_error=code;return 0;}
I put(std::shared_ptr<Resource> r) {
  std::lock_guard guard(registry_mutex);
  if(next_id==std::numeric_limits<I>::max())return failure(EOVERFLOW);
  const I id=next_id++;registry.emplace(id,std::move(r));return id;
}
template<class T> std::shared_ptr<T> get(I id,Kind kind) {
  std::lock_guard guard(registry_mutex);
  auto it=registry.find(id);
  if(it==registry.end()||it->second->kind!=kind){last_error=EBADF;return {};}
  return std::static_pointer_cast<T>(it->second);
}
void* start_thread(void* raw) {
  auto& t=*static_cast<Thread*>(raw);
  t.result=sounio_native_callback_thunk(t.callback,t.argument);
  return nullptr;
}
int wait_ready(int fd,short events,I timeout_ms) {
  if(timeout_ms<0||timeout_ms>2147483647){last_error=EINVAL;return -1;}
  using Clock=std::chrono::steady_clock;
  const auto end=Clock::now()+std::chrono::milliseconds(timeout_ms);
  for(;;) {
    const auto now=Clock::now();
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(end-now).count();
    const int ms=ns<=0?0:static_cast<int>((ns+999999)/1000000);
    pollfd p{fd,events,0};
    const int rc=::poll(&p,1,ms);
    if(rc>0){if(p.revents&POLLNVAL){last_error=EBADF;return -1;}return 0;}
    if(rc==0){last_error=ETIMEDOUT;return -1;}
    if(errno!=EINTR){last_error=errno;return -1;}
    if(Clock::now()>=end){last_error=ETIMEDOUT;return -1;}
  }
}
std::shared_ptr<Socket> adopt_socket(int fd) {
  if(fd<0){last_error=errno;return {};}
  try{return std::make_shared<Socket>(fd);}
  catch(...){::close(fd);throw;}
}
sockaddr_in address(I ip,I port) {
  sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(static_cast<std::uint32_t>(ip));
  a.sin_port=htons(static_cast<std::uint16_t>(port));return a;
}
bool valid_address(I ip,I port) {return ip>=0&&ip<=4294967295LL&&port>=0&&port<=65535;}
I dispatch(I op,I a,I b,I c,I d) {
  if(op==1)return 1;
  if(op==2) {
    std::lock_guard guard(registry_mutex);auto it=registry.find(a);
    if(it==registry.end())return failure(EBADF);
    if(it->second->kind==Kind::thread) {
      auto t=std::static_pointer_cast<Thread>(it->second);
      std::lock_guard thread_guard(t->mutex);
      if(!t->joined)return failure(EBUSY);
    }
    if(it->second->kind==Kind::mutex) {
      // The registry lock excludes a new lookup during final destruction.
      if(it->second.use_count()!=1)return failure(EBUSY);
      auto m=std::static_pointer_cast<Mutex>(it->second);
      if(m->users.load()!=0)return failure(EBUSY);
      const int rc=pthread_mutex_destroy(&m->native);if(rc)return failure(rc);
      m->initialized=false;
    }
    registry.erase(it);return 0;
  }
  if(op==3){std::lock_guard guard(registry_mutex);return static_cast<I>(registry.size());}
  if(op==10)return put(std::make_shared<Atomic>(a));
  if(op>=11&&op<=15) {
    auto r=get<Atomic>(a,Kind::atomic);if(!r)return 0;
    if(op==11)return r->value.load();
    if(op==12){r->value.store(b);return 0;}
    if(op==13)return r->value.fetch_add(b);
    if(op==15)return r->value.fetch_sub(b);
    return r->value.compare_exchange_strong(b,c)?1:0;
  }
  if(op==16) {
    auto m=std::make_shared<Mutex>();const int rc=m->initialize();
    if(rc)return failure(rc);
    return put(m);
  }
  if(op>=17&&op<=19) {
    auto m=get<Mutex>(a,Kind::mutex);if(!m)return 0;
    if(op==19) {
      const int rc=pthread_mutex_unlock(&m->native);if(rc)return failure(rc);
      m->users.fetch_sub(1);return 0;
    }
    m->users.fetch_add(1);
    const int rc=op==17?pthread_mutex_lock(&m->native):pthread_mutex_trylock(&m->native);
    if(rc){m->users.fetch_sub(1);return failure(rc);}
    return 1;
  }
  if(op==20) {
    if(std::getenv("SOUNIO_PLATFORM_TRACE"))std::fprintf(stderr,"PLATFORM_CALLBACK_ADDRESS=0x%llx ARG=%lld\n",static_cast<unsigned long long>(a),static_cast<long long>(b));
    Dl_info info{};
    if(a==0||dladdr(reinterpret_cast<void*>(a),&info)==0)return failure(EINVAL);
    auto t=std::make_shared<Thread>(reinterpret_cast<Thread::Callback>(a),b);
    // Insert before starting: a failed registry allocation cannot orphan a thread.
    const I id=put(t);if(!id)return 0;
    const int rc=pthread_create(&t->id,nullptr,start_thread,t.get());
    if(rc){std::lock_guard guard(registry_mutex);registry.erase(id);return failure(rc);}
    {std::lock_guard guard(t->mutex);t->started=true;}
    return id;
  }
  if(op==21) {
    auto t=get<Thread>(a,Kind::thread);if(!t)return 0;
    {std::lock_guard guard(t->mutex);
      if(!t->started||t->joining||t->joined)return failure(EINVAL);
      if(pthread_equal(pthread_self(),t->id))return failure(EDEADLK);
      t->joining=true;
    }
    const int rc=pthread_join(t->id,nullptr);
    {std::lock_guard guard(t->mutex);t->joining=false;t->joined=rc==0;}
    if(rc)return failure(rc);
    return t->result;
  }
  if(op==22){if(sched_yield()!=0)return failure(errno);return 0;}
  if(op==30) {
    timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))return failure(errno);
    if(t.tv_sec>(std::numeric_limits<I>::max()-t.tv_nsec)/1000000000)return failure(EOVERFLOW);
    return static_cast<I>(t.tv_sec)*1000000000+t.tv_nsec;
  }
  if(op==40) {
    if(b<0||(b>0&&a==0))return failure(EINVAL);
    I done=0;auto* p=reinterpret_cast<unsigned char*>(a);
    while(done<b) {
      const auto rc=getrandom(p+done,static_cast<std::size_t>(b-done),0);
      if(rc>0){done+=rc;continue;}
      if(rc<0&&errno==EINTR)continue;
      last_error=rc==0?EIO:errno;return done;
    }
    return done;
  }
  if(op==50||op==52||op==60) {
    if(!valid_address(a,b))return failure(EINVAL);
    if(op==50&&(c<=0||c>2147483647))return failure(EINVAL);
    if(op==52&&(c<0||c>2147483647))return failure(EINVAL);
    auto s=adopt_socket(::socket(AF_INET,(op==60?SOCK_DGRAM:SOCK_STREAM)|SOCK_CLOEXEC|SOCK_NONBLOCK,0));
    if(!s)return 0;
    auto addr=address(a,b);
    if(op==52) {
      if(::connect(s->fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0) {
        if(errno!=EINPROGRESS)return failure(errno);
        if(wait_ready(s->fd,POLLOUT,c))return 0;
        int err=0;socklen_t len=sizeof(err);
        if(getsockopt(s->fd,SOL_SOCKET,SO_ERROR,&err,&len))return failure(errno);
        if(err)return failure(err);
      }
    } else {
      int yes=1;
      if(setsockopt(s->fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes)))return failure(errno);
      if(::bind(s->fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr)))return failure(errno);
      if(op==50&&::listen(s->fd,static_cast<int>(c)))return failure(errno);
    }
    return put(s);
  }
  if(op>=51&&op<=63) {
    auto s=get<Socket>(a,Kind::socket);if(!s)return 0;
    if(op==51) {
      sockaddr_in addr{};socklen_t len=sizeof(addr);
      if(getsockname(s->fd,reinterpret_cast<sockaddr*>(&addr),&len))return failure(errno);
      return ntohs(addr.sin_port);
    }
    if(op==53) {
      if(wait_ready(s->fd,POLLIN,b))return 0;
      const int fd=accept4(s->fd,nullptr,nullptr,SOCK_CLOEXEC|SOCK_NONBLOCK);
      if(fd<0)return failure(errno);
      return put(adopt_socket(fd));
    }
    if(op==54||op==55||op==61||op==62) {
      if(c<0||(c>0&&b==0))return failure(EINVAL);
      if(op==61) {
        if(d<=0||d>65535)return failure(EINVAL);
        auto addr=address(2130706433,d);
        const auto rc=sendto(s->fd,reinterpret_cast<void*>(b),static_cast<std::size_t>(c),MSG_NOSIGNAL,reinterpret_cast<sockaddr*>(&addr),sizeof(addr));
        if(rc<0)return failure(errno);
      return rc;
      }
      // A zero-length TCP read/write completes immediately. UDP receive still
      // consumes a datagram, including a zero-length datagram.
      if(c==0&&op!=62)return 0;
      if(wait_ready(s->fd,op==54?POLLOUT:POLLIN,d))return 0;
      ssize_t rc=0;
      do {
        if(op==54)rc=::send(s->fd,reinterpret_cast<void*>(b),static_cast<std::size_t>(c),MSG_NOSIGNAL);
        else if(op==55)rc=::recv(s->fd,reinterpret_cast<void*>(b),static_cast<std::size_t>(c),0);
        else {sockaddr_in peer{};
          iovec buffer{reinterpret_cast<void*>(b),static_cast<std::size_t>(c)};
          msghdr message{};message.msg_name=&peer;message.msg_namelen=sizeof(peer);
          message.msg_iov=&buffer;message.msg_iovlen=1;
          rc=recvmsg(s->fd,&message,0);
          if(rc>=0) {
            s->peer_port.store(ntohs(peer.sin_port));
            // The datagram is consumed. Expose truncation and the copied byte
            // count; callers must not accept its prefix as a complete message.
            if(message.msg_flags&MSG_TRUNC){last_error=EMSGSIZE;return rc;}
          }
        }
      }while(rc<0&&errno==EINTR);
      if(rc<0)return failure(errno);
      return rc;
    }
    if(op==56){if(b<0||b>2)return failure(EINVAL);if(::shutdown(s->fd,static_cast<int>(b)))return failure(errno);return 0;}
    if(op==63)return s->peer_port.load();
  }
  return failure(ENOSYS);
}
}
extern "C" std::int64_t sounio_platform_call(std::int64_t op,std::int64_t a,std::int64_t b,std::int64_t c,std::int64_t d) noexcept {
  if(op==0)return last_error;
  last_error=0;
  try{return dispatch(op,a,b,c,d);}
  catch(const std::bad_alloc&){return failure(ENOMEM);}
  catch(...){return failure(EIO);}
}

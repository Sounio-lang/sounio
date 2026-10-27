// Minimal Linux platform ABI for the Sounio stdlib. Algorithms stay in Sounio.
// All entrypoints are noexcept. Operation 0 reads this thread's last errno;
// every other operation resets it. Resource IDs are never raw addresses.
#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <vector>
#include <string>
#include <cstring>
#include <fcntl.h>
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
enum class Kind {atomic,thread,socket,mutex,channel,file,text,pollset,pool};
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
struct Channel final:Resource {
  std::mutex mutex;
  std::condition_variable readable,writable;
  std::vector<I> data;
  std::size_t head=0,size=0;
  bool closed=false,destroyed=false;
  explicit Channel(std::size_t capacity):Resource(Kind::channel),data(capacity){}
};
I context_count=0; // protected by registry_mutex

struct Text final:Resource {
  std::vector<char> data;
  Text(const char* source,std::size_t size):Resource(Kind::text),data(size+1) {
    if(size>0)std::memcpy(data.data(),source,size);
  }
};
struct File final:Resource {
  int fd;std::mutex mutex;
  explicit File(int f):Resource(Kind::file),fd(f){}
  ~File()override {if(fd>=0)::close(fd);}
};
struct Socket final:Resource {
  int fd;std::atomic<I> peer_endpoint{0};
  std::mutex destination_mutex;sockaddr_in destination{};bool has_destination=false;
  std::mutex connect_mutex;bool is_connection=false,connect_pending=false;int connect_error=0;
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

struct PollRegistration {I socket,tag;short interests,ready=0;};
static_assert(sizeof(PollRegistration)==24);
struct PollSet final:Resource {
  std::mutex mutex;I capacity;std::vector<PollRegistration> entries;
  explicit PollSet(I n):Resource(Kind::pollset),capacity(n){entries.reserve(static_cast<std::size_t>(n));}
};
bool monotonic_now(I& value) {
  timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t)){last_error=errno;return false;}
  if(t.tv_sec>(std::numeric_limits<I>::max()-t.tv_nsec)/1000000000){last_error=EOVERFLOW;return false;}
  value=static_cast<I>(t.tv_sec)*1000000000+t.tv_nsec;return true;
}
// Return count, 0 on deadline, -1 on error. No registry lock is held here.
int poll_absolute(pollfd* descriptors,nfds_t count,I deadline,bool immediate=false) {
  if(deadline<0){last_error=EINVAL;return -1;}
  for(;;) {
    I now=0;if(!monotonic_now(now))return -1;
    const I remaining=deadline>now?deadline-now:0;
    const I rounded=remaining/1000000+(remaining%1000000!=0);
    const int milliseconds=deadline==0||immediate?0:static_cast<int>(std::min<I>(rounded,2147483647));
    const int rc=::poll(descriptors,count,milliseconds);
    if(rc>0)return rc;
    if(rc<0&&errno!=EINTR){last_error=errno;return -1;}
    if(deadline==0||immediate)return 0;
    if(!monotonic_now(now))return -1;
    if(now>=deadline)return 0;
  }
}
short ready_flags(short flags) {
  return static_cast<short>(((flags&POLLIN)?1:0)|((flags&POLLOUT)?2:0)|
    ((flags&POLLERR)?4:0)|((flags&POLLHUP)?8:0)|((flags&POLLNVAL)?16:0));
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
I channel_send(const std::shared_ptr<Channel>& q,I value,I timeout_ms,bool wait) {
  if(wait&&(timeout_ms<0||timeout_ms>2147483647))return failure(EINVAL);
  std::unique_lock lock(q->mutex);
  const auto ready=[&]{return q->destroyed||q->closed||q->size<q->data.size();};
  if(!ready()) {
    if(!wait)return failure(EAGAIN);
    if(!q->writable.wait_until(lock,std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms),ready))return failure(ETIMEDOUT);
  }
  if(q->destroyed)return failure(EBADF);
  if(q->closed)return failure(EPIPE);
  q->data[(q->head+q->size)%q->data.size()]=value;++q->size;
  lock.unlock();q->readable.notify_one();return 0;
}
I channel_receive(const std::shared_ptr<Channel>& q,I timeout_ms,bool wait) {
  if(wait&&(timeout_ms<0||timeout_ms>2147483647))return failure(EINVAL);
  std::unique_lock lock(q->mutex);
  const auto ready=[&]{return q->destroyed||q->closed||q->size>0;};
  if(!ready()) {
    if(!wait)return failure(EAGAIN);
    if(!q->readable.wait_until(lock,std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms),ready))return failure(ETIMEDOUT);
  }
  if(q->destroyed)return failure(EBADF);
  if(q->size==0)return failure(ENODATA);
  const I value=q->data[q->head];q->head=(q->head+1)%q->data.size();--q->size;
  lock.unlock();q->writable.notify_one();return value;
}


// Fixed worker count; every admitted task owns one bounded slot until collected.
// A task carries scalar data, never a Sounio owner, loan or closure environment.
struct Pool final:Resource {
  struct Slot {I id=0,argument=0,value=0;int state=0,error=0;};
  using Callback=I(*)(I);
  static constexpr std::size_t stack_bytes=1048576,guard_bytes=4096;
  Callback callback;
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<Slot> slots;
  std::vector<pthread_t> threads;
  I next_task=0,outstanding=0,running=0,completed=0,peak_running=0;
  bool accepting=true,joining=false,joined=false;
  Pool(Callback f,std::size_t capacity,std::size_t workers)
      :Resource(Kind::pool),callback(f),slots(capacity) {threads.reserve(workers);}
  ~Pool() override {
    // Construction failure is also joined before any callback/library can vanish.
    {std::lock_guard lock(mutex);accepting=false;}
    changed.notify_all();
    if(!joined)for(auto thread:threads)pthread_join(thread,nullptr);
  }
  static void* run(void* raw) noexcept {
    auto& pool=*static_cast<Pool*>(raw);
    for(;;) {
      std::unique_lock lock(pool.mutex);
      auto queued=[&] {return std::any_of(pool.slots.begin(),pool.slots.end(),
                               [](const Slot& s){return s.state==1;});};
      pool.changed.wait(lock,[&]{return !pool.accepting||queued();});
      std::size_t index=pool.slots.size();I smallest=std::numeric_limits<I>::max();
      for(std::size_t i=0;i<pool.slots.size();++i)
        if(pool.slots[i].state==1&&pool.slots[i].id<smallest){index=i;smallest=pool.slots[i].id;}
      if(index==pool.slots.size()){if(!pool.accepting)return nullptr;continue;}
      auto& slot=pool.slots[index];
      slot.state=2;++pool.running;pool.peak_running=std::max(pool.peak_running,pool.running);
      const I argument=slot.argument;
      lock.unlock();
      I value=0;int error=0;
      try {value=sounio_native_callback_thunk(pool.callback,argument);}
      catch(...) {error=EIO;}
      lock.lock();
      slot.value=value;slot.error=error;slot.state=3;
      --pool.running;++pool.completed;
      lock.unlock();pool.changed.notify_all();
    }
  }
  int start(std::size_t workers) {
    pthread_attr_t attr;
    int rc=pthread_attr_init(&attr);if(rc)return rc;
    rc=pthread_attr_setstacksize(&attr,stack_bytes);
    if(!rc)rc=pthread_attr_setguardsize(&attr,guard_bytes);
    if(rc){pthread_attr_destroy(&attr);return rc;}
    for(std::size_t i=0;i<workers;++i) {
      pthread_t thread{};
      rc=pthread_create(&thread,&attr,run,this);
      if(rc)break;
      threads.push_back(thread);
    }
    pthread_attr_destroy(&attr);
    if(rc) {
      {std::lock_guard lock(mutex);accepting=false;}
      changed.notify_all();
      for(auto thread:threads)pthread_join(thread,nullptr);
      joined=true;
    }
    return rc;
  }
  I stop(bool cancel) {
    std::lock_guard lock(mutex);accepting=false;
    I canceled=0;
    if(cancel)for(auto& s:slots)if(s.state==1){s.state=3;s.error=ECANCELED;++canceled;++completed;}
    changed.notify_all();return canceled;
  }
  int join() {
    {
      std::lock_guard lock(mutex);
      if(accepting||joining||joined)return EINVAL;
      for(auto thread:threads)if(pthread_equal(pthread_self(),thread))return EDEADLK;
      joining=true;
    }
    int error=0;
    for(auto thread:threads){const int rc=pthread_join(thread,nullptr);if(rc&&!error)error=rc;}
    {std::lock_guard lock(mutex);joining=false;joined=error==0;}
    changed.notify_all();return error;
  }
};
static_assert(sizeof(Pool)<=1024&&sizeof(Pool::Slot)<=48&&sizeof(pthread_t)<=8);
// Reservation includes dedicated 1MiB stacks + 4KiB guards and payload/control
// allowances. It excludes OS allocator bookkeeping and shared runtime pages.
I pool_reservation(I workers,I capacity) {
  if(workers<1||workers>64||capacity<1||capacity>1048576)return -1;
  return 1024+workers*(1048576+4096+8)+capacity*48;
}
I pool_submit(const std::shared_ptr<Pool>& pool,I argument,I timeout_ms) {
  if(timeout_ms<0||timeout_ms>2147483647)return failure(EINVAL);
  std::unique_lock lock(pool->mutex);
  const auto ready=[&]{return !pool->accepting||pool->outstanding<static_cast<I>(pool->slots.size());};
  if(!ready()) {
    if(timeout_ms==0)return failure(EAGAIN);
    if(!pool->changed.wait_until(lock,std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms),ready))
      return failure(ETIMEDOUT);
  }
  if(!pool->accepting)return failure(EPIPE);
  if(pool->next_task==std::numeric_limits<I>::max())return failure(EOVERFLOW);
  for(auto& slot:pool->slots)if(slot.state==0) {
    slot={pool->next_task++,argument,0,1,0};++pool->outstanding;
    const I id=slot.id;lock.unlock();pool->changed.notify_all();return id;
  }
  return failure(EIO);
}
I pool_result(const std::shared_ptr<Pool>& pool,I id,I timeout_ms) {
  if(id<0||timeout_ms<0||timeout_ms>2147483647)return failure(EINVAL);
  std::unique_lock lock(pool->mutex);
  auto found=std::find_if(pool->slots.begin(),pool->slots.end(),
                         [&](const Pool::Slot& s){return s.state!=0&&s.id==id;});
  if(found==pool->slots.end())return failure(ENOENT);
  const auto ready=[&]{return found->state==0||found->id!=id||found->state==3;};
  if(!ready()) {
    if(timeout_ms==0)return failure(EAGAIN);
    if(!pool->changed.wait_until(lock,std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms),ready))
      return failure(ETIMEDOUT);
  }
  if(found->state!=3||found->id!=id)return failure(ENOENT);
  const I value=found->value;const int error=found->error;
  found->state=0;--pool->outstanding;
  lock.unlock();pool->changed.notify_all();
  if(error)return failure(error);
  return value;
}

I dispatch(I op,I a,I b,I c,I d) {
  if(op==100) {
    if(a<=0||b<0)return failure(EINVAL);
    if(a>std::numeric_limits<I>::max()/24)return failure(EOVERFLOW);
    if(a*24>b)return failure(ENOBUFS);
    if(static_cast<std::uint64_t>(a)>std::vector<PollRegistration>{}.max_size())return failure(EOVERFLOW);
    return put(std::make_shared<PollSet>(a));
  }
  if(op>=101&&op<=106) {
    auto set=get<PollSet>(a,Kind::pollset);if(!set)return 0;
    std::lock_guard lock(set->mutex);
    if(op==101) {
      if(c<1||c>3)return failure(EINVAL);
      if(!get<Socket>(b,Kind::socket))return 0;
      auto found=std::find_if(set->entries.begin(),set->entries.end(),[&](const auto& e){return e.socket==b;});
      if(found==set->entries.end()) {
        if(set->entries.size()>=static_cast<std::size_t>(set->capacity))return failure(ENOBUFS);
        set->entries.push_back({b,d,static_cast<short>(c),0});
      } else {found->tag=d;found->interests=static_cast<short>(c);}
      for(auto& entry:set->entries)entry.ready=0;
      return 1;
    }
    if(op==102) {
      auto found=std::find_if(set->entries.begin(),set->entries.end(),[&](const auto& e){return e.socket==b;});
      if(found==set->entries.end())return failure(ENOENT);
      set->entries.erase(found);for(auto& entry:set->entries)entry.ready=0;return 1;
    }
    if(op==103) {
      if(b<0)return failure(EINVAL);
      std::vector<pollfd> descriptors;descriptors.reserve(set->entries.size());
      std::vector<std::shared_ptr<Socket>> owners;owners.reserve(set->entries.size());
      bool invalid=false;
      for(auto& entry:set->entries) {
        auto socket=get<Socket>(entry.socket,Kind::socket);
        entry.ready=socket?0:16;invalid|=!socket;last_error=0;
        descriptors.push_back({socket?socket->fd:-1,static_cast<short>(((entry.interests&1)?POLLIN:0)|((entry.interests&2)?POLLOUT:0)),0});
        owners.push_back(std::move(socket));
      }
      if(poll_absolute(descriptors.data(),descriptors.size(),b,invalid)<0)return 0;
      I count=0;
      for(std::size_t i=0;i<set->entries.size();++i) {
        auto current=get<Socket>(set->entries[i].socket,Kind::socket);
        set->entries[i].ready=!current?16:ready_flags(descriptors[i].revents);
        last_error=0;if(set->entries[i].ready)++count;
      }
      return count;
    }
    if(b<0)return failure(ERANGE);
    for(const auto& entry:set->entries)if(entry.ready) {
      if(b--==0)return op==104?entry.socket:op==105?entry.ready:entry.tag;
    }
    return failure(ERANGE);
  }
  if(op>=107&&op<=112) {
    auto socket=get<Socket>(a,Kind::socket);if(!socket)return 0;
    if(op==109)return socket->peer_endpoint.load()>>16;
    if(op==112)return socket->peer_endpoint.load();
    int type=0;socklen_t length=sizeof(type);
    if(getsockopt(socket->fd,SOL_SOCKET,SO_TYPE,&type,&length))return failure(errno);
    if(op==107) {
      if(type!=SOCK_DGRAM)return failure(EPROTOTYPE);
      if(!valid_address(b,c)||c==0)return failure(EINVAL);
      std::lock_guard lock(socket->destination_mutex);
      socket->destination=address(b,c);socket->has_destination=true;return 1;
    }
    if(c<0||(c>0&&b==0))return failure(EINVAL);
    if(op==108) {
      if(type!=SOCK_DGRAM)return failure(EPROTOTYPE);
      sockaddr_in destination{};
      {std::lock_guard lock(socket->destination_mutex);
       if(!socket->has_destination)return failure(EDESTADDRREQ);
       destination=socket->destination;}
      ssize_t transferred;
      do {transferred=sendto(socket->fd,reinterpret_cast<const void*>(b),static_cast<std::size_t>(c),MSG_NOSIGNAL,reinterpret_cast<const sockaddr*>(&destination),sizeof(destination));}while(transferred<0&&errno==EINTR);
      return transferred<0?failure(errno):transferred;
    }
    if(type!=SOCK_STREAM)return failure(EPROTOTYPE);
    if(d<0)return failure(EINVAL);
    if(c==0)return 0;
    bool retried=false;
    for(;;) {
      I now=0;if(!monotonic_now(now))return 0;
      if(retried&&(d==0||now>=d))return failure(ETIMEDOUT);
      pollfd descriptor{socket->fd,static_cast<short>(op==110?POLLOUT:POLLIN),0};
      const int ready=poll_absolute(&descriptor,1,d);
      if(ready<0)return 0;
      if(ready==0)return failure(ETIMEDOUT);
      if(descriptor.revents&POLLNVAL)return failure(EBADF);
      const ssize_t transferred=op==110?
        ::send(socket->fd,reinterpret_cast<const void*>(b),static_cast<std::size_t>(c),MSG_NOSIGNAL):
        ::recv(socket->fd,reinterpret_cast<void*>(b),static_cast<std::size_t>(c),0);
      if(transferred>=0)return transferred;
      if(errno!=EINTR&&errno!=EAGAIN&&errno!=EWOULDBLOCK)return failure(errno);
      retried=true;
    }
  }

  if(op==139) {
    const I reserved=pool_reservation(a,b);
    if(reserved<0)return failure(EINVAL);
    return reserved;
  }
  if(op==130) {
    const I reserved=pool_reservation(b,c);
    if(reserved<0||d<0||a==0)return failure(EINVAL);
    if(d<reserved)return failure(ENOBUFS);
    // dladdr rejects heap closure descriptors. It does not prove code identity;
    // the typed public API requires a bare function and forbids owner casts.
    Dl_info info{};
    if(dladdr(reinterpret_cast<void*>(a),&info)==0)return failure(EINVAL);
    auto pool=std::make_shared<Pool>(reinterpret_cast<Pool::Callback>(a),
                                    static_cast<std::size_t>(c),static_cast<std::size_t>(b));
    const int rc=pool->start(static_cast<std::size_t>(b));if(rc)return failure(rc);
    return put(std::move(pool));
  }
  if(op>=131&&op<=138) {
    auto pool=get<Pool>(a,Kind::pool);if(!pool)return 0;
    if(op==131){if(d!=0)return failure(EINVAL);return pool_submit(pool,b,c);}
    if(op==132){if(d!=0)return failure(EINVAL);return pool_result(pool,b,c);}
    if(op==133){if(b<0||b>1||c!=0||d!=0)return failure(EINVAL);return pool->stop(b==1);}
    if(op==134){if(b||c||d)return failure(EINVAL);const int rc=pool->join();return rc?failure(rc):0;}
    if(op==135) {
      std::lock_guard lock(pool->mutex);
      if(c||d)return failure(EINVAL);
      if(b==0)return pool->outstanding;
      if(b==1)return static_cast<I>(pool->threads.size());
      if(b==2)return static_cast<I>(pool->slots.size());
      if(b==3)return pool->running;
      if(b==4)return pool->peak_running;
      if(b==5)return pool->completed;
      return failure(EINVAL);
    }
    if(op==136) {
      std::lock_guard lock(pool->mutex);
      if(b||c||d||!pool->joined)return failure(EINVAL);
      const I discarded=pool->outstanding;
      for(auto& slot:pool->slots)slot.state=0;
      pool->outstanding=0;return discarded;
    }
    return failure(ENOSYS);
  }

  if(op==95) {
    if(b<0||c<1||(b>0&&a==0))return failure(EINVAL);
    if(b>c-1)return failure(ENOBUFS);
    if(static_cast<std::uint64_t>(b)+1>std::vector<char>{}.max_size())return failure(EOVERFLOW);
    const auto* data=reinterpret_cast<const char*>(a);
    for(I i=0;i<b;++i)if(data[i]==0)return failure(EILSEQ);
    return put(std::make_shared<Text>(data,static_cast<std::size_t>(b)));
  }
  if(op==96) {
    auto t=get<Text>(a,Kind::text);if(!t)return 0;
    return reinterpret_cast<I>(t->data.data());
  }

  if(op==90) {
    if(a==0||b<=0||b>4095||c<0||c>2)return failure(EINVAL);
    const auto* path=reinterpret_cast<const char*>(a);
    for(I i=0;i<b;++i)if(path[i]==0)return failure(EINVAL);
    const std::string owned(path,static_cast<std::size_t>(b));
    const int flags=O_CLOEXEC|(c==0?O_RDONLY:(O_WRONLY|O_CREAT|(c==1?O_TRUNC:O_APPEND)));
    int fd;
    do {fd=::open(owned.c_str(),flags,0600);}while(fd<0&&errno==EINTR);
    if(fd<0)return failure(errno);
    std::shared_ptr<File> owned_file;
    try{owned_file=std::make_shared<File>(fd);}catch(...){::close(fd);throw;}
    return put(std::move(owned_file));
  }
  if(op>=91&&op<=94) {
    auto f=get<File>(a,Kind::file);if(!f)return 0;
    std::lock_guard lock(f->mutex);
    if(op==91||op==92) {
      if(c<0||(c>0&&b==0))return failure(EINVAL);
      ssize_t rc;
      do {rc=op==91?::read(f->fd,reinterpret_cast<void*>(b),static_cast<std::size_t>(c))
                    : ::write(f->fd,reinterpret_cast<const void*>(b),static_cast<std::size_t>(c));}while(rc<0&&errno==EINTR);
      if(rc<0)return failure(errno);
      return rc;
    }
    if(op==93) {
      if(c<0||c>2)return failure(EINVAL);
      const auto rc=::lseek(f->fd,static_cast<off_t>(b),static_cast<int>(c));
      if(rc<0)return failure(errno);
      return static_cast<I>(rc);
    }
    int rc;do {rc=::fsync(f->fd);}while(rc<0&&errno==EINTR);
    if(rc<0)return failure(errno);
    return 0;
  }

  if(op==80) {
    std::lock_guard guard(registry_mutex);
    if(context_count==std::numeric_limits<I>::max())return failure(EOVERFLOW);
    return ++context_count;
  }
  if(op==81) {
    std::lock_guard guard(registry_mutex);
    if(context_count<=0)return failure(EINVAL);
    if(context_count==1&&!registry.empty())return failure(EBUSY);
    --context_count;return 0;
  }
  if(op==70) {
    if(a<=0||b<0)return failure(EINVAL);
    if(a>b/8)return failure(ENOBUFS);
    if(static_cast<std::uint64_t>(a)>std::vector<I>{}.max_size())return failure(EOVERFLOW);
    return put(std::make_shared<Channel>(static_cast<std::size_t>(a)));
  }
  if(op>=71&&op<=77) {
    auto q=get<Channel>(a,Kind::channel);if(!q)return 0;
    if(op==71||op==73)return channel_send(q,b,c,op==73);
    if(op==72||op==74)return channel_receive(q,b,op==74);
    if(op==75){
      {std::lock_guard lock(q->mutex);q->closed=true;}
      q->readable.notify_all();q->writable.notify_all();return 0;
    }
    if(op==76)return 1;
    std::lock_guard lock(q->mutex);return static_cast<I>(q->size);
  }

  if(op==1)return 1;
  if(op==2) {
    std::lock_guard guard(registry_mutex);auto it=registry.find(a);
    if(it==registry.end())return failure(EBADF);
    int close_error=0;
    if(it->second->kind==Kind::pool) {
      auto pool=std::static_pointer_cast<Pool>(it->second);
      std::lock_guard lock(pool->mutex);
      if(!pool->joined||pool->outstanding!=0)return failure(EBUSY);
    }
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
    if(it->second->kind==Kind::channel) {
      auto q=std::static_pointer_cast<Channel>(it->second);
      {std::lock_guard lock(q->mutex);q->closed=true;q->destroyed=true;}
      q->readable.notify_all();q->writable.notify_all();
    }
    if(it->second->kind==Kind::file) {
      auto f=std::static_pointer_cast<File>(it->second);
      std::lock_guard file_guard(f->mutex);
      // Linux consumes the descriptor even when close reports EINTR or delayed I/O.
      // Clear ownership first: neither a retry nor the destructor may close a reused fd.
      const int fd=f->fd;f->fd=-1;
      if(::close(fd)!=0)close_error=errno;
    }
    registry.erase(it);
    return close_error!=0?failure(close_error):0;
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
  if(op==50||op==52||op==59||op==60) {
    if(!valid_address(a,b))return failure(EINVAL);
    if(op==50&&(c<=0||c>2147483647))return failure(EINVAL);
    if(op==52&&(c<0||c>2147483647))return failure(EINVAL);
    auto s=adopt_socket(::socket(AF_INET,(op==60?SOCK_DGRAM:SOCK_STREAM)|SOCK_CLOEXEC|SOCK_NONBLOCK,0));
    if(!s)return 0;
    auto addr=address(a,b);
    if(op==52||op==59) {
      s->is_connection=true;
      if(::connect(s->fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0) {
        if(errno!=EINPROGRESS)return failure(errno);
        if(op==59){s->connect_pending=true;return put(s);}
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
  if(op>=51&&op<=64) {
    auto s=get<Socket>(a,Kind::socket);if(!s)return 0;
    if(op==51) {
      sockaddr_in addr{};socklen_t len=sizeof(addr);
      if(getsockname(s->fd,reinterpret_cast<sockaddr*>(&addr),&len))return failure(errno);
      return ntohs(addr.sin_port);
    }
    if(op==57) {
      if(b<1||b>3||c!=0||d!=0)return failure(EINVAL);
      pollfd ready{s->fd,static_cast<short>(((b&1)?POLLIN:0)|((b&2)?POLLOUT:0)),0};
      int rc;do {rc=::poll(&ready,1,0);}while(rc<0&&errno==EINTR);
      if(rc<0)return failure(errno);
      if(ready.revents&POLLNVAL)return failure(EBADF);
      return ((ready.revents&POLLIN)?1:0)|((ready.revents&POLLOUT)?2:0)|((ready.revents&POLLERR)?4:0)|((ready.revents&POLLHUP)?8:0);
    }
    if(op==58){int type=0;socklen_t length=sizeof(type);if(getsockopt(s->fd,SOL_SOCKET,SO_TYPE,&type,&length))return failure(errno);return type==SOCK_DGRAM?2:1;}
    if(op==64) {
      std::lock_guard lock(s->connect_mutex);
      if(!s->is_connection)return failure(ENOTCONN);
      if(s->connect_error)return failure(s->connect_error);
      if(!s->connect_pending)return 0;
      pollfd ready{s->fd,POLLOUT,0};
      int rc;do {rc=::poll(&ready,1,0);}while(rc<0&&errno==EINTR);
      if(rc<0)return failure(errno);
      if(rc==0)return failure(EAGAIN);
      int err=0;socklen_t length=sizeof(err);
      if(getsockopt(s->fd,SOL_SOCKET,SO_ERROR,&err,&length))return failure(errno);
      s->connect_error=err;s->connect_pending=false;
      if(err)return failure(err);
      return 0;
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
            s->peer_endpoint.store((static_cast<I>(ntohl(peer.sin_addr.s_addr))<<16)|ntohs(peer.sin_port));
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
    if(op==63)return s->peer_endpoint.load()&65535;
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

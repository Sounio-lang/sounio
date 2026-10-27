#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <initializer_list>
#include <unistd.h>
using I=std::int64_t;
extern "C" I sounio_platform_call(I,I,I,I,I) noexcept;
extern "C" int __real_close(int);
static int injected_error=0,close_calls=0,closed_fd=-1;
extern "C" int __wrap_close(int fd) {
 ++close_calls;closed_fd=fd;
 const int rc=__real_close(fd);
 if(injected_error!=0){const int code=injected_error;injected_error=0;errno=code;return -1;}
 return rc;
}
static I call(I op,I a=0,I b=0,I c=0,I d=0){return sounio_platform_call(op,a,b,c,d);}
int main(){
 int failures=0;
 const char* path="native-file-close-reference.tmp";
 ::unlink(path);
 for(const int expected:{0,EIO,EINTR}){
  const I owner=call(90,reinterpret_cast<I>(path),std::strlen(path),1);
  if(owner<=0||call(0)!=0)return 2;
  const int before=close_calls;injected_error=expected;
  const I result=call(2,owner),status=call(0);
  if(result!=0||status!=expected||close_calls!=before+1){
   std::printf("CLOSE_STATUS expected=%d actual=%lld calls=%d\n",expected,static_cast<long long>(status),close_calls-before);++failures;
  }
  if(call(3)!=0)return 3;
  // A failed close still consumes the Linux fd. Never retry a potentially reused fd.
  const int reused=::open(path,O_RDONLY|O_CLOEXEC);
  if(reused<0||reused!=closed_fd)return 4;
  if(call(2,owner)!=0||call(0)!=EBADF||::fcntl(reused,F_GETFD)<0)return 5;
  if(::close(reused)!=0)return 6;
 }
 ::unlink(path);
 if(failures!=0)return 1;
 std::puts("NATIVE_FILE_CLOSE_CPP_OK");
}


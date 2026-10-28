#define _LARGEFILE64_SOURCE 1
#include <sys/stat.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#if !defined(__linux__) || !defined(__x86_64__)
#error This oracle qualifies only the admitted Linux x86-64 host ABI.
#endif
static_assert(offsetof(struct stat64,st_dev)==0);
static_assert(offsetof(struct stat64,st_ino)==8);
static_assert(sizeof(struct stat64)<=256);
static_assert(sizeof(dev_t)==8 && sizeof(ino64_t)==8);
int main(int argc,char**argv){
 if(argc!=3)return 64;
 struct stat64 a{},b{};
 if(stat64(argv[1],&a)||stat64(argv[2],&b))return 1;
 std::uint64_t words[2]{};
 std::memcpy(words,&a,sizeof words);
 if(words[0]!=a.st_dev||words[1]!=a.st_ino||a.st_ino==0)return 2;
 if(a.st_dev!=b.st_dev||a.st_ino!=b.st_ino)return 3;
 std::cout<<"stat64 layout and physical identity: OK\n";
}

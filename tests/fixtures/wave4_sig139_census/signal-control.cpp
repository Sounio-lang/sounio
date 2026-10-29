#include <csignal>
int main(){return std::raise(SIGSEGV);}

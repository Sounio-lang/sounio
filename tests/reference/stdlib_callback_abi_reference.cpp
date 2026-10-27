#include <cassert>
#include <cstdint>
#include <iostream>
using I=std::int64_t;
using Callback=I(*)(I);
using Adapter=I(*)(Callback,I);
extern "C" I sounio_native_callback_thunk(Callback,I);
extern "C" I unguarded_adapter(Callback,I);
extern "C" I register_probe(Adapter);
asm(
".text\n"
".type clobber_callback,@function\n"
"clobber_callback:\n"
"movq %rsp,%rax\n"
"andq $15,%rax\n"
"cmpq $8,%rax\n"
"jne 9f\n"
"leaq 7(%rdi),%rax\n"
"jmp 8f\n"
"9: movq $-1,%rax\n"
"8: movq $0,%rbp\n"
"movq $0,%rbx\n"
"movq $0,%r12\n"
"movq $0,%r13\n"
"movq $0,%r14\n"
"movq $0,%r15\n"
"ret\n"
".globl unguarded_adapter\n"
"unguarded_adapter:\n"
"movq %rdi,%rax\n"
"movq %rsi,%rdi\n"
"jmp *%rax\n"
".globl register_probe\n"
"register_probe:\n"
"pushq %rbp\n"
"pushq %rbx\n"
"pushq %r12\n"
"pushq %r13\n"
"pushq %r14\n"
"pushq %r15\n"
"subq $24,%rsp\n"
"movq %rdi,8(%rsp)\n"
"movq $11,%rbp\n"
"movq $12,%rbx\n"
"movq $13,%r12\n"
"movq $14,%r13\n"
"movq $15,%r14\n"
"movq $16,%r15\n"
"leaq clobber_callback(%rip),%rdi\n"
"movq $35,%rsi\n"
"call *8(%rsp)\n"
"xorq %rdx,%rdx\n"
"cmpq $42,%rax\n"
"je 1f\n"
"orq $64,%rdx\n"
"1: cmpq $11,%rbp\n"
"je 2f\n"
"orq $1,%rdx\n"
"2: cmpq $12,%rbx\n"
"je 3f\n"
"orq $2,%rdx\n"
"3: cmpq $13,%r12\n"
"je 4f\n"
"orq $4,%rdx\n"
"4: cmpq $14,%r13\n"
"je 5f\n"
"orq $8,%rdx\n"
"5: cmpq $15,%r14\n"
"je 6f\n"
"orq $16,%rdx\n"
"6: cmpq $16,%r15\n"
"je 7f\n"
"orq $32,%rdx\n"
"7: movq %rdx,%rax\n"
"addq $24,%rsp\n"
"popq %r15\n"
"popq %r14\n"
"popq %r13\n"
"popq %r12\n"
"popq %rbx\n"
"popq %rbp\n"
"ret\n"
);
int main(){
 assert(register_probe(&unguarded_adapter)==63);
 assert(register_probe(&sounio_native_callback_thunk)==0);
 std::cout<<"CALLBACK_SYSV_SIX_REGISTERS_ALIGNMENT_RETURN_OK\n";
}


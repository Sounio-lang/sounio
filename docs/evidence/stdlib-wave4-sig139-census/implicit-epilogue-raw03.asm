 5a73f20:	48 89 85 40 51 f6 ff 	mov    %rax,-0x9aec0(%rbp)
 5a73f27:	48 8b 82 30 00 00 00 	mov    0x30(%rdx),%rax
 5a73f2e:	48 89 85 48 51 f6 ff 	mov    %rax,-0x9aeb8(%rbp)
 5a73f35:	48 8b 82 38 00 00 00 	mov    0x38(%rdx),%rax
 5a73f3c:	48 89 85 50 51 f6 ff 	mov    %rax,-0x9aeb0(%rbp)
 5a73f43:	48 8b 82 40 00 00 00 	mov    0x40(%rdx),%rax
 5a73f4a:	48 89 85 58 51 f6 ff 	mov    %rax,-0x9aea8(%rbp)
 5a73f51:	48 8b 82 48 00 00 00 	mov    0x48(%rdx),%rax
 5a73f58:	48 89 85 60 51 f6 ff 	mov    %rax,-0x9aea0(%rbp)
 5a73f5f:	48 8b 82 50 00 00 00 	mov    0x50(%rdx),%rax
 5a73f66:	48 89 85 68 51 f6 ff 	mov    %rax,-0x9ae98(%rbp)
 5a73f6d:	48 8b 82 58 00 00 00 	mov    0x58(%rdx),%rax
 5a73f74:	48 89 85 70 51 f6 ff 	mov    %rax,-0x9ae90(%rbp)
 5a73f7b:	48 8b 82 60 00 00 00 	mov    0x60(%rdx),%rax
 5a73f82:	48 89 85 78 51 f6 ff 	mov    %rax,-0x9ae88(%rbp)
 5a73f89:	48 8b 82 68 00 00 00 	mov    0x68(%rdx),%rax
 5a73f90:	48 89 85 80 51 f6 ff 	mov    %rax,-0x9ae80(%rbp)
 5a73f97:	48 8b 82 70 00 00 00 	mov    0x70(%rdx),%rax
 5a73f9e:	48 89 85 88 51 f6 ff 	mov    %rax,-0x9ae78(%rbp)
 5a73fa5:	48 8b 82 78 00 00 00 	mov    0x78(%rdx),%rax
 5a73fac:	48 89 85 90 51 f6 ff 	mov    %rax,-0x9ae70(%rbp)
 5a73fb3:	48 8d 85 18 51 f6 ff 	lea    -0x9aee8(%rbp),%rax
 5a73fba:	50                   	push   %rax
 5a73fbb:	48 8b b4 24 08 00 00 	mov    0x8(%rsp),%rsi
 5a73fc2:	00 
 5a73fc3:	48 8b 94 24 00 00 00 	mov    0x0(%rsp),%rdx
 5a73fca:	00 
 5a73fcb:	48 8d 85 08 48 f6 ff 	lea    -0x9b7f8(%rbp),%rax
 5a73fd2:	48 89 c7             	mov    %rax,%rdi
 5a73fd5:	e8 0a cc f3 ff       	call   0x59b0be4
 5a73fda:	48 83 c4 10          	add    $0x10,%rsp
 5a73fde:	48 89 c2             	mov    %rax,%rdx
 5a73fe1:	48 8b 85 28 d3 f6 ff 	mov    -0x92cd8(%rbp),%rax
 5a73fe8:	48 89 d6             	mov    %rdx,%rsi
 5a73feb:	48 89 c7             	mov    %rax,%rdi
 5a73fee:	b9 22 01 00 00       	mov    $0x122,%ecx
 5a73ff3:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 5a73ff6:	48 8b 85 28 d3 f6 ff 	mov    -0x92cd8(%rbp),%rax
 5a73ffd:	e9 07 00 00 00       	jmp    0x5a74009
 5a74002:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 5a74009:	31 c0                	xor    %eax,%eax
 5a7400b:	48 81 c4 00 b8 09 00 	add    $0x9b800,%rsp
 5a74012:	4c 89 e0             	mov    %r12,%rax
 5a74015:	41 5c                	pop    %r12
 5a74017:	5d                   	pop    %rbp
 5a74018:	c3                   	ret
 5a74019:	55                   	push   %rbp

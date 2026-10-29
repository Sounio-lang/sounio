 5a68ef2:	48 8b 82 50 00 00 00 	mov    0x50(%rdx),%rax
 5a68ef9:	48 89 85 a8 e3 fd ff 	mov    %rax,-0x21c58(%rbp)
 5a68f00:	48 8b 82 58 00 00 00 	mov    0x58(%rdx),%rax
 5a68f07:	48 89 85 b0 e3 fd ff 	mov    %rax,-0x21c50(%rbp)
 5a68f0e:	48 8b 82 60 00 00 00 	mov    0x60(%rdx),%rax
 5a68f15:	48 89 85 b8 e3 fd ff 	mov    %rax,-0x21c48(%rbp)
 5a68f1c:	48 8b 82 68 00 00 00 	mov    0x68(%rdx),%rax
 5a68f23:	48 89 85 c0 e3 fd ff 	mov    %rax,-0x21c40(%rbp)
 5a68f2a:	48 8b 82 70 00 00 00 	mov    0x70(%rdx),%rax
 5a68f31:	48 89 85 c8 e3 fd ff 	mov    %rax,-0x21c38(%rbp)
 5a68f38:	48 8b 82 78 00 00 00 	mov    0x78(%rdx),%rax
 5a68f3f:	48 89 85 d0 e3 fd ff 	mov    %rax,-0x21c30(%rbp)
 5a68f46:	48 8d 85 58 e3 fd ff 	lea    -0x21ca8(%rbp),%rax
 5a68f4d:	50                   	push   %rax
 5a68f4e:	48 8b b4 24 08 00 00 	mov    0x8(%rsp),%rsi
 5a68f55:	00 
 5a68f56:	48 8b 94 24 00 00 00 	mov    0x0(%rsp),%rdx
 5a68f5d:	00 
 5a68f5e:	48 8d 85 48 da fd ff 	lea    -0x225b8(%rbp),%rax
 5a68f65:	48 89 c7             	mov    %rax,%rdi
 5a68f68:	e8 77 7c f4 ff       	call   0x59b0be4
 5a68f6d:	48 83 c4 10          	add    $0x10,%rsp
 5a68f71:	48 89 c2             	mov    %rax,%rdx
 5a68f74:	4c 89 e1             	mov    %r12,%rcx
 5a68f77:	48 89 d6             	mov    %rdx,%rsi
 5a68f7a:	48 89 cf             	mov    %rcx,%rdi
 5a68f7d:	b9 22 01 00 00       	mov    $0x122,%ecx
 5a68f82:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 5a68f85:	4c 89 e0             	mov    %r12,%rax
 5a68f88:	48 81 c4 00 b8 09 00 	add    $0x9b800,%rsp
 5a68f8f:	4c 89 e0             	mov    %r12,%rax
 5a68f92:	41 5c                	pop    %r12
 5a68f94:	5d                   	pop    %rbp
 5a68f95:	c3                   	ret
 5a68f96:	e9 c9 00 00 00       	jmp    0x5a69064
 5a68f9b:	48 8b 85 88 77 fe ff 	mov    -0x18878(%rbp),%rax
 5a68fa2:	48 8b 00             	mov    (%rax),%rax

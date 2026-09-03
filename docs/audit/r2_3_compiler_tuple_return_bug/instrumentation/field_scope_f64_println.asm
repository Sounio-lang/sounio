    786e:	48 89 e5             	mov    %rsp,%rbp
    7871:	48 81 ec b0 01 00 00 	sub    $0x1b0,%rsp
    7878:	48 89 f8             	mov    %rdi,%rax
    787b:	48 89 85 f8 ff ff ff 	mov    %rax,-0x8(%rbp)
    7882:	48 89 f0             	mov    %rsi,%rax
    7885:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
    788c:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    7893:	00 00 00 
    7896:	48 8b 00             	mov    (%rax),%rax
    7899:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
    78a0:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
    78a7:	50                   	push   %rax
    78a8:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
    78af:	59                   	pop    %rcx
    78b0:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    78b5:	66 48 0f 6e c8       	movq   %rax,%xmm1
    78ba:	66 0f 2e c1          	ucomisd %xmm1,%xmm0
    78be:	0f 95 c0             	setne  %al
    78c1:	48 0f b6 c0          	movzbq %al,%rax
    78c5:	48 85 c0             	test   %rax,%rax
    78c8:	0f 84 33 00 00 00    	je     0x7901
    78ce:	48 b8 6e 61 6e 00 00 	movabs $0x6e616e,%rax
    78d5:	00 00 00 
    78d8:	50                   	push   %rax
    78d9:	b8 01 00 00 00       	mov    $0x1,%eax
    78de:	bf 01 00 00 00       	mov    $0x1,%edi
    78e3:	48 89 e6             	mov    %rsp,%rsi
    78e6:	ba 03 00 00 00       	mov    $0x3,%edx
    78eb:	0f 05                	syscall
    78ed:	48 81 c4 08 00 00 00 	add    $0x8,%rsp
    78f4:	31 c0                	xor    %eax,%eax
    78f6:	31 c0                	xor    %eax,%eax
    78f8:	48 81 c4 b0 01 00 00 	add    $0x1b0,%rsp
    78ff:	5d                   	pop    %rbp
    7900:	c3                   	ret
    7901:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
    7908:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
    790f:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    7916:	50                   	push   %rax
    7917:	31 c0                	xor    %eax,%eax
    7919:	59                   	pop    %rcx
    791a:	48 39 c1             	cmp    %rax,%rcx
    791d:	0f 9c c0             	setl   %al
    7920:	48 0f b6 c0          	movzbq %al,%rax
    7924:	48 85 c0             	test   %rax,%rax
    7927:	0f 84 09 00 00 00    	je     0x7936
    792d:	31 c0                	xor    %eax,%eax
    792f:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
    7936:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    793d:	50                   	push   %rax
    793e:	48 b8 0c 00 00 00 00 	movabs $0xc,%rax
    7945:	00 00 00 
    7948:	59                   	pop    %rcx
    7949:	48 39 c1             	cmp    %rax,%rcx
    794c:	0f 9f c0             	setg   %al
    794f:	48 0f b6 c0          	movzbq %al,%rax
    7953:	48 85 c0             	test   %rax,%rax
    7956:	0f 84 11 00 00 00    	je     0x796d
    795c:	48 b8 0c 00 00 00 00 	movabs $0xc,%rax
    7963:	00 00 00 
    7966:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
    796d:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
    7974:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
    797b:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
    7982:	48 89 85 d0 ff ff ff 	mov    %rax,-0x30(%rbp)
    7989:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7990:	50                   	push   %rax
    7991:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7998:	00 00 00 
    799b:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    79a0:	48 b8 00 00 00 00 00 	movabs $0x4024000000000000,%rax
    79a7:	00 24 40 
    79aa:	66 48 0f 6e c8       	movq   %rax,%xmm1
    79af:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
    79b3:	66 48 0f 7e c0       	movq   %xmm0,%rax
    79b8:	59                   	pop    %rcx
    79b9:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    79be:	66 48 0f 6e c8       	movq   %rax,%xmm1
    79c3:	66 0f 2e c1          	ucomisd %xmm1,%xmm0
    79c7:	0f 92 c0             	setb   %al
    79ca:	48 0f b6 c0          	movzbq %al,%rax
    79ce:	48 85 c0             	test   %rax,%rax
    79d1:	0f 84 91 00 00 00    	je     0x7a68
    79d7:	48 b8 2d 00 00 00 00 	movabs $0x2d,%rax
    79de:	00 00 00 
    79e1:	48 83 ec 08          	sub    $0x8,%rsp
    79e5:	88 04 24             	mov    %al,(%rsp)
    79e8:	b8 01 00 00 00       	mov    $0x1,%eax
    79ed:	bf 01 00 00 00       	mov    $0x1,%edi
    79f2:	48 89 e6             	mov    %rsp,%rsi
    79f5:	ba 01 00 00 00       	mov    $0x1,%edx
    79fa:	0f 05                	syscall
    79fc:	48 83 c4 08          	add    $0x8,%rsp
    7a00:	31 c0                	xor    %eax,%eax
    7a02:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7a09:	00 00 00 
    7a0c:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7a11:	48 b8 00 00 00 00 00 	movabs $0x4024000000000000,%rax
    7a18:	00 24 40 
    7a1b:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7a20:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
    7a24:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7a29:	50                   	push   %rax
    7a2a:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7a31:	59                   	pop    %rcx
    7a32:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7a37:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7a3c:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
    7a40:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7a45:	48 89 85 c8 ff ff ff 	mov    %rax,-0x38(%rbp)
    7a4c:	48 8b 85 c8 ff ff ff 	mov    -0x38(%rbp),%rax
    7a53:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
    7a5a:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
    7a61:	48 89 85 d0 ff ff ff 	mov    %rax,-0x30(%rbp)
    7a68:	31 c0                	xor    %eax,%eax
    7a6a:	48 89 85 c0 ff ff ff 	mov    %rax,-0x40(%rbp)
    7a71:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    7a78:	00 00 00 
    7a7b:	48 89 85 b8 ff ff ff 	mov    %rax,-0x48(%rbp)
    7a82:	48 8b 85 c0 ff ff ff 	mov    -0x40(%rbp),%rax
    7a89:	50                   	push   %rax
    7a8a:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    7a91:	59                   	pop    %rcx
    7a92:	48 39 c1             	cmp    %rax,%rcx
    7a95:	0f 9c c0             	setl   %al
    7a98:	48 0f b6 c0          	movzbq %al,%rax
    7a9c:	48 85 c0             	test   %rax,%rax
    7a9f:	0f 84 43 00 00 00    	je     0x7ae8
    7aa5:	48 8b 85 b8 ff ff ff 	mov    -0x48(%rbp),%rax
    7aac:	50                   	push   %rax
    7aad:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    7ab4:	00 00 00 
    7ab7:	59                   	pop    %rcx
    7ab8:	48 87 c1             	xchg   %rax,%rcx
    7abb:	48 0f af c1          	imul   %rcx,%rax
    7abf:	48 89 85 b8 ff ff ff 	mov    %rax,-0x48(%rbp)
    7ac6:	48 8b 85 c0 ff ff ff 	mov    -0x40(%rbp),%rax
    7acd:	50                   	push   %rax
    7ace:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    7ad5:	00 00 00 
    7ad8:	59                   	pop    %rcx
    7ad9:	48 01 c8             	add    %rcx,%rax
    7adc:	48 89 85 c0 ff ff ff 	mov    %rax,-0x40(%rbp)
    7ae3:	e9 9a ff ff ff       	jmp    0x7a82
    7ae8:	48 8b 85 b8 ff ff ff 	mov    -0x48(%rbp),%rax
    7aef:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7af4:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7af9:	48 89 85 b0 ff ff ff 	mov    %rax,-0x50(%rbp)
    7b00:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7b07:	50                   	push   %rax
    7b08:	48 b8 00 80 c6 a4 7e 	movabs $0x38d7ea4c68000,%rax
    7b0f:	8d 03 00 
    7b12:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7b17:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7b1c:	59                   	pop    %rcx
    7b1d:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7b22:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7b27:	66 0f 2e c1          	ucomisd %xmm1,%xmm0
    7b2b:	0f 93 c0             	setae  %al
    7b2e:	48 0f b6 c0          	movzbq %al,%rax
    7b32:	48 85 c0             	test   %rax,%rax
    7b35:	0f 84 f7 03 00 00    	je     0x7f32
    7b3b:	31 c0                	xor    %eax,%eax
    7b3d:	48 89 85 a8 ff ff ff 	mov    %rax,-0x58(%rbp)
    7b44:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7b4b:	50                   	push   %rax
    7b4c:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    7b53:	00 00 00 
    7b56:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7b5b:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7b60:	59                   	pop    %rcx
    7b61:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7b66:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7b6b:	66 0f 2e c1          	ucomisd %xmm1,%xmm0
    7b6f:	0f 93 c0             	setae  %al
    7b72:	48 0f b6 c0          	movzbq %al,%rax
    7b76:	48 85 c0             	test   %rax,%rax
    7b79:	0f 84 78 00 00 00    	je     0x7bf7
    7b7f:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7b86:	48 89 85 a0 ff ff ff 	mov    %rax,-0x60(%rbp)
    7b8d:	50                   	push   %rax
    7b8e:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    7b95:	00 00 00 
    7b98:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7b9d:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7ba2:	48 89 85 98 ff ff ff 	mov    %rax,-0x68(%rbp)
    7ba9:	59                   	pop    %rcx
    7baa:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7baf:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7bb4:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
    7bb8:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7bbd:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
    7bc4:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7bcb:	00 00 00 
    7bce:	48 89 85 d0 ff ff ff 	mov    %rax,-0x30(%rbp)
    7bd5:	48 8b 85 a8 ff ff ff 	mov    -0x58(%rbp),%rax
    7bdc:	50                   	push   %rax
    7bdd:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    7be4:	00 00 00 
    7be7:	59                   	pop    %rcx
    7be8:	48 01 c8             	add    %rcx,%rax
    7beb:	48 89 85 a8 ff ff ff 	mov    %rax,-0x58(%rbp)
    7bf2:	e9 4d ff ff ff       	jmp    0x7b44
    7bf7:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7bfe:	66 48 0f 6e c0       	movq   %rax,%xmm0
    7c03:	f2 48 0f 2c c0       	cvttsd2si %xmm0,%rax
    7c08:	48 89 85 90 ff ff ff 	mov    %rax,-0x70(%rbp)
    7c0f:	48 8b 85 90 ff ff ff 	mov    -0x70(%rbp),%rax
    7c16:	50                   	push   %rax
    7c17:	51                   	push   %rcx
    7c18:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
    7c1f:	48 89 c1             	mov    %rax,%rcx
    7c22:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    7c29:	00 00 00 
    7c2c:	48 89 08             	mov    %rcx,(%rax)
    7c2f:	48 89 c8             	mov    %rcx,%rax
    7c32:	59                   	pop    %rcx
    7c33:	48 8b bc 24 00 00 00 	mov    0x0(%rsp),%rdi
    7c3a:	00 
    7c3b:	e8 da f8 ff ff       	call   0x751a
    7c40:	48 83 c4 08          	add    $0x8,%rsp
    7c44:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    7c4b:	50                   	push   %rax
    7c4c:	31 c0                	xor    %eax,%eax
    7c4e:	59                   	pop    %rcx
    7c4f:	48 39 c1             	cmp    %rax,%rcx
    7c52:	0f 9f c0             	setg   %al
    7c55:	48 0f b6 c0          	movzbq %al,%rax
    7c59:	48 85 c0             	test   %rax,%rax
    7c5c:	0f 84 62 02 00 00    	je     0x7ec4
    7c62:	48 b8 2e 00 00 00 00 	movabs $0x2e,%rax
    7c69:	00 00 00 
    7c6c:	48 83 ec 08          	sub    $0x8,%rsp
    7c70:	88 04 24             	mov    %al,(%rsp)
    7c73:	b8 01 00 00 00       	mov    $0x1,%eax
    7c78:	bf 01 00 00 00       	mov    $0x1,%edi
    7c7d:	48 89 e6             	mov    %rsp,%rsi
    7c80:	ba 01 00 00 00       	mov    $0x1,%edx
    7c85:	0f 05                	syscall
    7c87:	48 83 c4 08          	add    $0x8,%rsp
    7c8b:	31 c0                	xor    %eax,%eax
    7c8d:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7c94:	50                   	push   %rax
    7c95:	48 8b 85 90 ff ff ff 	mov    -0x70(%rbp),%rax
    7c9c:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7ca1:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7ca6:	59                   	pop    %rcx
    7ca7:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7cac:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7cb1:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
    7cb5:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7cba:	48 89 85 88 ff ff ff 	mov    %rax,-0x78(%rbp)
    7cc1:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7cc8:	00 00 00 
    7ccb:	48 89 85 80 ff ff ff 	mov    %rax,-0x80(%rbp)
    7cd2:	48 8b 85 88 ff ff ff 	mov    -0x78(%rbp),%rax
    7cd9:	48 89 85 78 ff ff ff 	mov    %rax,-0x88(%rbp)
    7ce0:	50                   	push   %rax
    7ce1:	48 8b 85 b0 ff ff ff 	mov    -0x50(%rbp),%rax
    7ce8:	48 89 85 70 ff ff ff 	mov    %rax,-0x90(%rbp)
    7cef:	59                   	pop    %rcx
    7cf0:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7cf5:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7cfa:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    7cfe:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7d03:	48 89 85 68 ff ff ff 	mov    %rax,-0x98(%rbp)
    7d0a:	48 8b 85 78 ff ff ff 	mov    -0x88(%rbp),%rax
    7d11:	50                   	push   %rax
    7d12:	48 8b 85 78 ff ff ff 	mov    -0x88(%rbp),%rax
    7d19:	59                   	pop    %rcx
    7d1a:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7d1f:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7d24:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    7d28:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7d2d:	50                   	push   %rax
    7d2e:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7d35:	00 00 00 
    7d38:	59                   	pop    %rcx
    7d39:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7d3e:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7d43:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    7d47:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7d4c:	50                   	push   %rax
    7d4d:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
    7d54:	50                   	push   %rax
    7d55:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
    7d5c:	59                   	pop    %rcx
    7d5d:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7d62:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7d67:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    7d6b:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7d70:	50                   	push   %rax
    7d71:	48 8b 85 80 ff ff ff 	mov    -0x80(%rbp),%rax
    7d78:	59                   	pop    %rcx
    7d79:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7d7e:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7d83:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    7d87:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7d8c:	59                   	pop    %rcx
    7d8d:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7d92:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7d97:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    7d9b:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7da0:	48 89 85 60 ff ff ff 	mov    %rax,-0xa0(%rbp)
    7da7:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    7dae:	50                   	push   %rax
    7daf:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7db6:	00 00 00 
    7db9:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7dbe:	48 b8 05 00 00 00 00 	movabs $0x5,%rax
    7dc5:	00 00 00 
    7dc8:	f2 48 0f 2a c8       	cvtsi2sd %rax,%xmm1
    7dcd:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    7dd4:	00 00 00 
    7dd7:	f2 48 0f 2a d0       	cvtsi2sd %rax,%xmm2
    7ddc:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
    7de0:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    7de4:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7de9:	59                   	pop    %rcx
    7dea:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7def:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7df4:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    7df8:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7dfd:	48 89 85 58 ff ff ff 	mov    %rax,-0xa8(%rbp)
    7e04:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
    7e0b:	66 48 0f 6e c0       	movq   %rax,%xmm0
    7e10:	f2 48 0f 2c c0       	cvttsd2si %xmm0,%rax
    7e15:	48 89 85 50 ff ff ff 	mov    %rax,-0xb0(%rbp)
    7e1c:	48 8b 85 50 ff ff ff 	mov    -0xb0(%rbp),%rax
    7e23:	50                   	push   %rax
    7e24:	48 8b 85 b8 ff ff ff 	mov    -0x48(%rbp),%rax
    7e2b:	59                   	pop    %rcx
    7e2c:	48 39 c1             	cmp    %rax,%rcx
    7e2f:	0f 9d c0             	setge  %al
    7e32:	48 0f b6 c0          	movzbq %al,%rax
    7e36:	48 85 c0             	test   %rax,%rax
    7e39:	0f 84 20 00 00 00    	je     0x7e5f
    7e3f:	48 8b 85 b8 ff ff ff 	mov    -0x48(%rbp),%rax
    7e46:	50                   	push   %rax
    7e47:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    7e4e:	00 00 00 
    7e51:	59                   	pop    %rcx
    7e52:	48 87 c1             	xchg   %rax,%rcx
    7e55:	48 29 c8             	sub    %rcx,%rax
    7e58:	48 89 85 50 ff ff ff 	mov    %rax,-0xb0(%rbp)
    7e5f:	48 8b 85 50 ff ff ff 	mov    -0xb0(%rbp),%rax
    7e66:	50                   	push   %rax
    7e67:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    7e6e:	50                   	push   %rax
    7e6f:	51                   	push   %rcx
    7e70:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7e77:	00 00 00 
    7e7a:	48 89 c1             	mov    %rax,%rcx
    7e7d:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    7e84:	00 00 00 
    7e87:	48 89 08             	mov    %rcx,(%rax)
    7e8a:	48 89 c8             	mov    %rcx,%rax
    7e8d:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7e94:	00 00 00 
    7e97:	48 89 c1             	mov    %rax,%rcx
    7e9a:	48 b8 a8 00 00 10 00 	movabs $0x100000a8,%rax
    7ea1:	00 00 00 
    7ea4:	48 89 08             	mov    %rcx,(%rax)
    7ea7:	48 89 c8             	mov    %rcx,%rax
    7eaa:	59                   	pop    %rcx
    7eab:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
    7eb2:	00 
    7eb3:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
    7eba:	00 
    7ebb:	e8 07 f8 ff ff       	call   0x76c7
    7ec0:	48 83 c4 10          	add    $0x10,%rsp
    7ec4:	48 b8 65 00 00 00 00 	movabs $0x65,%rax
    7ecb:	00 00 00 
    7ece:	48 83 ec 08          	sub    $0x8,%rsp
    7ed2:	88 04 24             	mov    %al,(%rsp)
    7ed5:	b8 01 00 00 00       	mov    $0x1,%eax
    7eda:	bf 01 00 00 00       	mov    $0x1,%edi
    7edf:	48 89 e6             	mov    %rsp,%rsi
    7ee2:	ba 01 00 00 00       	mov    $0x1,%edx
    7ee7:	0f 05                	syscall
    7ee9:	48 83 c4 08          	add    $0x8,%rsp
    7eed:	31 c0                	xor    %eax,%eax
    7eef:	48 8b 85 a8 ff ff ff 	mov    -0x58(%rbp),%rax
    7ef6:	50                   	push   %rax
    7ef7:	51                   	push   %rcx
    7ef8:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7eff:	00 00 00 
    7f02:	48 89 c1             	mov    %rax,%rcx
    7f05:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    7f0c:	00 00 00 
    7f0f:	48 89 08             	mov    %rcx,(%rax)
    7f12:	48 89 c8             	mov    %rcx,%rax
    7f15:	59                   	pop    %rcx
    7f16:	48 8b bc 24 00 00 00 	mov    0x0(%rsp),%rdi
    7f1d:	00 
    7f1e:	e8 f7 f5 ff ff       	call   0x751a
    7f23:	48 83 c4 08          	add    $0x8,%rsp
    7f27:	31 c0                	xor    %eax,%eax
    7f29:	48 81 c4 b0 01 00 00 	add    $0x1b0,%rsp
    7f30:	5d                   	pop    %rbp
    7f31:	c3                   	ret
    7f32:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7f39:	00 00 00 
    7f3c:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7f41:	48 b8 05 00 00 00 00 	movabs $0x5,%rax
    7f48:	00 00 00 
    7f4b:	f2 48 0f 2a c8       	cvtsi2sd %rax,%xmm1
    7f50:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    7f57:	00 00 00 
    7f5a:	f2 48 0f 2a d0       	cvtsi2sd %rax,%xmm2
    7f5f:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
    7f63:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    7f67:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7f6c:	48 89 85 48 ff ff ff 	mov    %rax,-0xb8(%rbp)
    7f73:	50                   	push   %rax
    7f74:	48 8b 85 b0 ff ff ff 	mov    -0x50(%rbp),%rax
    7f7b:	48 89 85 40 ff ff ff 	mov    %rax,-0xc0(%rbp)
    7f82:	59                   	pop    %rcx
    7f83:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7f88:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7f8d:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
    7f91:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7f96:	48 89 85 38 ff ff ff 	mov    %rax,-0xc8(%rbp)
    7f9d:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7fa4:	50                   	push   %rax
    7fa5:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    7fac:	00 00 00 
    7faf:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    7fb4:	48 b8 00 00 00 00 00 	movabs $0x4024000000000000,%rax
    7fbb:	00 24 40 
    7fbe:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7fc3:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
    7fc7:	66 48 0f 7e c0       	movq   %xmm0,%rax
    7fcc:	59                   	pop    %rcx
    7fcd:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    7fd2:	66 48 0f 6e c8       	movq   %rax,%xmm1
    7fd7:	66 0f 2e c1          	ucomisd %xmm1,%xmm0
    7fdb:	0f 97 c0             	seta   %al
    7fde:	48 0f b6 c0          	movzbq %al,%rax
    7fe2:	48 85 c0             	test   %rax,%rax
    7fe5:	0f 84 34 00 00 00    	je     0x801f
    7feb:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    7ff2:	50                   	push   %rax
    7ff3:	48 8b 85 38 ff ff ff 	mov    -0xc8(%rbp),%rax
    7ffa:	59                   	pop    %rcx
    7ffb:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8000:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8005:	66 0f 2e c1          	ucomisd %xmm1,%xmm0
    8009:	0f 92 c0             	setb   %al
    800c:	48 0f b6 c0          	movzbq %al,%rax
    8010:	48 85 c0             	test   %rax,%rax
    8013:	0f 95 c0             	setne  %al
    8016:	48 0f b6 c0          	movzbq %al,%rax
    801a:	e9 02 00 00 00       	jmp    0x8021
    801f:	31 c0                	xor    %eax,%eax
    8021:	48 85 c0             	test   %rax,%rax
    8024:	0f 84 ca 04 00 00    	je     0x84f4
    802a:	31 c0                	xor    %eax,%eax
    802c:	48 89 85 30 ff ff ff 	mov    %rax,-0xd0(%rbp)
    8033:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    803a:	50                   	push   %rax
    803b:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    8042:	00 00 00 
    8045:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    804a:	66 48 0f 7e c0       	movq   %xmm0,%rax
    804f:	59                   	pop    %rcx
    8050:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8055:	66 48 0f 6e c8       	movq   %rax,%xmm1
    805a:	66 0f 2e c1          	ucomisd %xmm1,%xmm0
    805e:	0f 92 c0             	setb   %al
    8061:	48 0f b6 c0          	movzbq %al,%rax
    8065:	48 85 c0             	test   %rax,%rax
    8068:	0f 84 20 01 00 00    	je     0x818e
    806e:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    8075:	48 89 85 28 ff ff ff 	mov    %rax,-0xd8(%rbp)
    807c:	50                   	push   %rax
    807d:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    8084:	00 00 00 
    8087:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    808c:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8091:	48 89 85 20 ff ff ff 	mov    %rax,-0xe0(%rbp)
    8098:	59                   	pop    %rcx
    8099:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    809e:	66 48 0f 6e c8       	movq   %rax,%xmm1
    80a3:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    80a7:	66 48 0f 7e c0       	movq   %xmm0,%rax
    80ac:	48 89 85 18 ff ff ff 	mov    %rax,-0xe8(%rbp)
    80b3:	48 8b 85 28 ff ff ff 	mov    -0xd8(%rbp),%rax
    80ba:	50                   	push   %rax
    80bb:	48 8b 85 28 ff ff ff 	mov    -0xd8(%rbp),%rax
    80c2:	59                   	pop    %rcx
    80c3:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    80c8:	66 48 0f 6e c8       	movq   %rax,%xmm1
    80cd:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    80d1:	66 48 0f 7e c0       	movq   %xmm0,%rax
    80d6:	50                   	push   %rax
    80d7:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    80de:	00 00 00 
    80e1:	59                   	pop    %rcx
    80e2:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    80e7:	66 48 0f 6e c8       	movq   %rax,%xmm1
    80ec:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    80f0:	66 48 0f 7e c0       	movq   %xmm0,%rax
    80f5:	50                   	push   %rax
    80f6:	48 8b 85 20 ff ff ff 	mov    -0xe0(%rbp),%rax
    80fd:	50                   	push   %rax
    80fe:	48 8b 85 20 ff ff ff 	mov    -0xe0(%rbp),%rax
    8105:	59                   	pop    %rcx
    8106:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    810b:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8110:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    8114:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8119:	50                   	push   %rax
    811a:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
    8121:	59                   	pop    %rcx
    8122:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8127:	66 48 0f 6e c8       	movq   %rax,%xmm1
    812c:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    8130:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8135:	59                   	pop    %rcx
    8136:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    813b:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8140:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    8144:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8149:	48 89 85 10 ff ff ff 	mov    %rax,-0xf0(%rbp)
    8150:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    8157:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
    815e:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    8165:	48 89 85 d0 ff ff ff 	mov    %rax,-0x30(%rbp)
    816c:	48 8b 85 30 ff ff ff 	mov    -0xd0(%rbp),%rax
    8173:	50                   	push   %rax
    8174:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    817b:	00 00 00 
    817e:	59                   	pop    %rcx
    817f:	48 01 c8             	add    %rcx,%rax
    8182:	48 89 85 30 ff ff ff 	mov    %rax,-0xd0(%rbp)
    8189:	e9 a5 fe ff ff       	jmp    0x8033
    818e:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    8195:	66 48 0f 6e c0       	movq   %rax,%xmm0
    819a:	f2 48 0f 2c c0       	cvttsd2si %xmm0,%rax
    819f:	48 89 85 08 ff ff ff 	mov    %rax,-0xf8(%rbp)
    81a6:	48 8b 85 08 ff ff ff 	mov    -0xf8(%rbp),%rax
    81ad:	50                   	push   %rax
    81ae:	51                   	push   %rcx
    81af:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
    81b6:	48 89 c1             	mov    %rax,%rcx
    81b9:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    81c0:	00 00 00 
    81c3:	48 89 08             	mov    %rcx,(%rax)
    81c6:	48 89 c8             	mov    %rcx,%rax
    81c9:	59                   	pop    %rcx
    81ca:	48 8b bc 24 00 00 00 	mov    0x0(%rsp),%rdi
    81d1:	00 
    81d2:	e8 43 f3 ff ff       	call   0x751a
    81d7:	48 83 c4 08          	add    $0x8,%rsp
    81db:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    81e2:	50                   	push   %rax
    81e3:	31 c0                	xor    %eax,%eax
    81e5:	59                   	pop    %rcx
    81e6:	48 39 c1             	cmp    %rax,%rcx
    81e9:	0f 9f c0             	setg   %al
    81ec:	48 0f b6 c0          	movzbq %al,%rax
    81f0:	48 85 c0             	test   %rax,%rax
    81f3:	0f 84 62 02 00 00    	je     0x845b
    81f9:	48 b8 2e 00 00 00 00 	movabs $0x2e,%rax
    8200:	00 00 00 
    8203:	48 83 ec 08          	sub    $0x8,%rsp
    8207:	88 04 24             	mov    %al,(%rsp)
    820a:	b8 01 00 00 00       	mov    $0x1,%eax
    820f:	bf 01 00 00 00       	mov    $0x1,%edi
    8214:	48 89 e6             	mov    %rsp,%rsi
    8217:	ba 01 00 00 00       	mov    $0x1,%edx
    821c:	0f 05                	syscall
    821e:	48 83 c4 08          	add    $0x8,%rsp
    8222:	31 c0                	xor    %eax,%eax
    8224:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    822b:	50                   	push   %rax
    822c:	48 8b 85 08 ff ff ff 	mov    -0xf8(%rbp),%rax
    8233:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    8238:	66 48 0f 7e c0       	movq   %xmm0,%rax
    823d:	59                   	pop    %rcx
    823e:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8243:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8248:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
    824c:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8251:	48 89 85 00 ff ff ff 	mov    %rax,-0x100(%rbp)
    8258:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    825f:	00 00 00 
    8262:	48 89 85 f8 fe ff ff 	mov    %rax,-0x108(%rbp)
    8269:	48 8b 85 00 ff ff ff 	mov    -0x100(%rbp),%rax
    8270:	48 89 85 f0 fe ff ff 	mov    %rax,-0x110(%rbp)
    8277:	50                   	push   %rax
    8278:	48 8b 85 b0 ff ff ff 	mov    -0x50(%rbp),%rax
    827f:	48 89 85 e8 fe ff ff 	mov    %rax,-0x118(%rbp)
    8286:	59                   	pop    %rcx
    8287:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    828c:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8291:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    8295:	66 48 0f 7e c0       	movq   %xmm0,%rax
    829a:	48 89 85 e0 fe ff ff 	mov    %rax,-0x120(%rbp)
    82a1:	48 8b 85 f0 fe ff ff 	mov    -0x110(%rbp),%rax
    82a8:	50                   	push   %rax
    82a9:	48 8b 85 f0 fe ff ff 	mov    -0x110(%rbp),%rax
    82b0:	59                   	pop    %rcx
    82b1:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    82b6:	66 48 0f 6e c8       	movq   %rax,%xmm1
    82bb:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    82bf:	66 48 0f 7e c0       	movq   %xmm0,%rax
    82c4:	50                   	push   %rax
    82c5:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    82cc:	00 00 00 
    82cf:	59                   	pop    %rcx
    82d0:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    82d5:	66 48 0f 6e c8       	movq   %rax,%xmm1
    82da:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    82de:	66 48 0f 7e c0       	movq   %xmm0,%rax
    82e3:	50                   	push   %rax
    82e4:	48 8b 85 e8 fe ff ff 	mov    -0x118(%rbp),%rax
    82eb:	50                   	push   %rax
    82ec:	48 8b 85 e8 fe ff ff 	mov    -0x118(%rbp),%rax
    82f3:	59                   	pop    %rcx
    82f4:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    82f9:	66 48 0f 6e c8       	movq   %rax,%xmm1
    82fe:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    8302:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8307:	50                   	push   %rax
    8308:	48 8b 85 f8 fe ff ff 	mov    -0x108(%rbp),%rax
    830f:	59                   	pop    %rcx
    8310:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8315:	66 48 0f 6e c8       	movq   %rax,%xmm1
    831a:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    831e:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8323:	59                   	pop    %rcx
    8324:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8329:	66 48 0f 6e c8       	movq   %rax,%xmm1
    832e:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    8332:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8337:	48 89 85 d8 fe ff ff 	mov    %rax,-0x128(%rbp)
    833e:	48 8b 85 e0 fe ff ff 	mov    -0x120(%rbp),%rax
    8345:	50                   	push   %rax
    8346:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    834d:	00 00 00 
    8350:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    8355:	48 b8 05 00 00 00 00 	movabs $0x5,%rax
    835c:	00 00 00 
    835f:	f2 48 0f 2a c8       	cvtsi2sd %rax,%xmm1
    8364:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    836b:	00 00 00 
    836e:	f2 48 0f 2a d0       	cvtsi2sd %rax,%xmm2
    8373:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
    8377:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    837b:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8380:	59                   	pop    %rcx
    8381:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8386:	66 48 0f 6e c8       	movq   %rax,%xmm1
    838b:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    838f:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8394:	48 89 85 d0 fe ff ff 	mov    %rax,-0x130(%rbp)
    839b:	48 8b 85 d0 fe ff ff 	mov    -0x130(%rbp),%rax
    83a2:	66 48 0f 6e c0       	movq   %rax,%xmm0
    83a7:	f2 48 0f 2c c0       	cvttsd2si %xmm0,%rax
    83ac:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    83b3:	48 8b 85 c8 fe ff ff 	mov    -0x138(%rbp),%rax
    83ba:	50                   	push   %rax
    83bb:	48 8b 85 b8 ff ff ff 	mov    -0x48(%rbp),%rax
    83c2:	59                   	pop    %rcx
    83c3:	48 39 c1             	cmp    %rax,%rcx
    83c6:	0f 9d c0             	setge  %al
    83c9:	48 0f b6 c0          	movzbq %al,%rax
    83cd:	48 85 c0             	test   %rax,%rax
    83d0:	0f 84 20 00 00 00    	je     0x83f6
    83d6:	48 8b 85 b8 ff ff ff 	mov    -0x48(%rbp),%rax
    83dd:	50                   	push   %rax
    83de:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    83e5:	00 00 00 
    83e8:	59                   	pop    %rcx
    83e9:	48 87 c1             	xchg   %rax,%rcx
    83ec:	48 29 c8             	sub    %rcx,%rax
    83ef:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    83f6:	48 8b 85 c8 fe ff ff 	mov    -0x138(%rbp),%rax
    83fd:	50                   	push   %rax
    83fe:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    8405:	50                   	push   %rax
    8406:	51                   	push   %rcx
    8407:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    840e:	00 00 00 
    8411:	48 89 c1             	mov    %rax,%rcx
    8414:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    841b:	00 00 00 
    841e:	48 89 08             	mov    %rcx,(%rax)
    8421:	48 89 c8             	mov    %rcx,%rax
    8424:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    842b:	00 00 00 
    842e:	48 89 c1             	mov    %rax,%rcx
    8431:	48 b8 a8 00 00 10 00 	movabs $0x100000a8,%rax
    8438:	00 00 00 
    843b:	48 89 08             	mov    %rcx,(%rax)
    843e:	48 89 c8             	mov    %rcx,%rax
    8441:	59                   	pop    %rcx
    8442:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
    8449:	00 
    844a:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
    8451:	00 
    8452:	e8 70 f2 ff ff       	call   0x76c7
    8457:	48 83 c4 10          	add    $0x10,%rsp
    845b:	48 b8 65 00 00 00 00 	movabs $0x65,%rax
    8462:	00 00 00 
    8465:	48 83 ec 08          	sub    $0x8,%rsp
    8469:	88 04 24             	mov    %al,(%rsp)
    846c:	b8 01 00 00 00       	mov    $0x1,%eax
    8471:	bf 01 00 00 00       	mov    $0x1,%edi
    8476:	48 89 e6             	mov    %rsp,%rsi
    8479:	ba 01 00 00 00       	mov    $0x1,%edx
    847e:	0f 05                	syscall
    8480:	48 83 c4 08          	add    $0x8,%rsp
    8484:	31 c0                	xor    %eax,%eax
    8486:	48 b8 2d 00 00 00 00 	movabs $0x2d,%rax
    848d:	00 00 00 
    8490:	48 83 ec 08          	sub    $0x8,%rsp
    8494:	88 04 24             	mov    %al,(%rsp)
    8497:	b8 01 00 00 00       	mov    $0x1,%eax
    849c:	bf 01 00 00 00       	mov    $0x1,%edi
    84a1:	48 89 e6             	mov    %rsp,%rsi
    84a4:	ba 01 00 00 00       	mov    $0x1,%edx
    84a9:	0f 05                	syscall
    84ab:	48 83 c4 08          	add    $0x8,%rsp
    84af:	31 c0                	xor    %eax,%eax
    84b1:	48 8b 85 30 ff ff ff 	mov    -0xd0(%rbp),%rax
    84b8:	50                   	push   %rax
    84b9:	51                   	push   %rcx
    84ba:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    84c1:	00 00 00 
    84c4:	48 89 c1             	mov    %rax,%rcx
    84c7:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    84ce:	00 00 00 
    84d1:	48 89 08             	mov    %rcx,(%rax)
    84d4:	48 89 c8             	mov    %rcx,%rax
    84d7:	59                   	pop    %rcx
    84d8:	48 8b bc 24 00 00 00 	mov    0x0(%rsp),%rdi
    84df:	00 
    84e0:	e8 35 f0 ff ff       	call   0x751a
    84e5:	48 83 c4 08          	add    $0x8,%rsp
    84e9:	31 c0                	xor    %eax,%eax
    84eb:	48 81 c4 b0 01 00 00 	add    $0x1b0,%rsp
    84f2:	5d                   	pop    %rbp
    84f3:	c3                   	ret
    84f4:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    84fb:	50                   	push   %rax
    84fc:	51                   	push   %rcx
    84fd:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
    8504:	48 89 c1             	mov    %rax,%rcx
    8507:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    850e:	00 00 00 
    8511:	48 89 08             	mov    %rcx,(%rax)
    8514:	48 89 c8             	mov    %rcx,%rax
    8517:	59                   	pop    %rcx
    8518:	48 8b bc 24 00 00 00 	mov    0x0(%rsp),%rdi
    851f:	00 
    8520:	e8 61 98 ff ff       	call   0x1d86
    8525:	48 83 c4 08          	add    $0x8,%rsp
    8529:	48 89 85 c0 fe ff ff 	mov    %rax,-0x140(%rbp)
    8530:	48 b8 98 00 00 10 00 	movabs $0x10000098,%rax
    8537:	00 00 00 
    853a:	48 8b 00             	mov    (%rax),%rax
    853d:	48 89 85 b8 fe ff ff 	mov    %rax,-0x148(%rbp)
    8544:	48 8b 85 b8 fe ff ff 	mov    -0x148(%rbp),%rax
    854b:	48 85 c0             	test   %rax,%rax
    854e:	0f 85 0c 00 00 00    	jne    0x8560
    8554:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
    855b:	e9 07 00 00 00       	jmp    0x8567
    8560:	48 8b 85 b8 fe ff ff 	mov    -0x148(%rbp),%rax
    8567:	48 89 85 b0 fe ff ff 	mov    %rax,-0x150(%rbp)
    856e:	48 8b 85 c0 fe ff ff 	mov    -0x140(%rbp),%rax
    8575:	48 89 85 a8 fe ff ff 	mov    %rax,-0x158(%rbp)
    857c:	48 8b 85 a8 fe ff ff 	mov    -0x158(%rbp),%rax
    8583:	66 48 0f 6e c0       	movq   %rax,%xmm0
    8588:	f2 48 0f 2c c0       	cvttsd2si %xmm0,%rax
    858d:	48 89 85 a0 fe ff ff 	mov    %rax,-0x160(%rbp)
    8594:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    859b:	50                   	push   %rax
    859c:	48 8b 85 a8 fe ff ff 	mov    -0x158(%rbp),%rax
    85a3:	59                   	pop    %rcx
    85a4:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    85a9:	66 48 0f 6e c8       	movq   %rax,%xmm1
    85ae:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
    85b2:	66 48 0f 7e c0       	movq   %xmm0,%rax
    85b7:	48 89 85 98 fe ff ff 	mov    %rax,-0x168(%rbp)
    85be:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
    85c5:	50                   	push   %rax
    85c6:	48 8b 85 b0 fe ff ff 	mov    -0x150(%rbp),%rax
    85cd:	59                   	pop    %rcx
    85ce:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    85d3:	66 48 0f 6e c8       	movq   %rax,%xmm1
    85d8:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    85dc:	66 48 0f 7e c0       	movq   %xmm0,%rax
    85e1:	48 89 85 90 fe ff ff 	mov    %rax,-0x170(%rbp)
    85e8:	48 8b 85 98 fe ff ff 	mov    -0x168(%rbp),%rax
    85ef:	48 89 85 88 fe ff ff 	mov    %rax,-0x178(%rbp)
    85f6:	50                   	push   %rax
    85f7:	48 8b 85 b0 ff ff ff 	mov    -0x50(%rbp),%rax
    85fe:	48 89 85 80 fe ff ff 	mov    %rax,-0x180(%rbp)
    8605:	59                   	pop    %rcx
    8606:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    860b:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8610:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    8614:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8619:	48 89 85 78 fe ff ff 	mov    %rax,-0x188(%rbp)
    8620:	48 8b 85 88 fe ff ff 	mov    -0x178(%rbp),%rax
    8627:	50                   	push   %rax
    8628:	48 8b 85 88 fe ff ff 	mov    -0x178(%rbp),%rax
    862f:	59                   	pop    %rcx
    8630:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8635:	66 48 0f 6e c8       	movq   %rax,%xmm1
    863a:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    863e:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8643:	50                   	push   %rax
    8644:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    864b:	00 00 00 
    864e:	59                   	pop    %rcx
    864f:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8654:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8659:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    865d:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8662:	50                   	push   %rax
    8663:	48 8b 85 80 fe ff ff 	mov    -0x180(%rbp),%rax
    866a:	50                   	push   %rax
    866b:	48 8b 85 80 fe ff ff 	mov    -0x180(%rbp),%rax
    8672:	59                   	pop    %rcx
    8673:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8678:	66 48 0f 6e c8       	movq   %rax,%xmm1
    867d:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    8681:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8686:	50                   	push   %rax
    8687:	48 8b 85 90 fe ff ff 	mov    -0x170(%rbp),%rax
    868e:	59                   	pop    %rcx
    868f:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8694:	66 48 0f 6e c8       	movq   %rax,%xmm1
    8699:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    869d:	66 48 0f 7e c0       	movq   %xmm0,%rax
    86a2:	59                   	pop    %rcx
    86a3:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    86a8:	66 48 0f 6e c8       	movq   %rax,%xmm1
    86ad:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    86b1:	66 48 0f 7e c0       	movq   %xmm0,%rax
    86b6:	48 89 85 70 fe ff ff 	mov    %rax,-0x190(%rbp)
    86bd:	48 8b 85 78 fe ff ff 	mov    -0x188(%rbp),%rax
    86c4:	50                   	push   %rax
    86c5:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    86cc:	00 00 00 
    86cf:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    86d4:	48 b8 05 00 00 00 00 	movabs $0x5,%rax
    86db:	00 00 00 
    86de:	f2 48 0f 2a c8       	cvtsi2sd %rax,%xmm1
    86e3:	48 b8 0a 00 00 00 00 	movabs $0xa,%rax
    86ea:	00 00 00 
    86ed:	f2 48 0f 2a d0       	cvtsi2sd %rax,%xmm2
    86f2:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
    86f6:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    86fa:	66 48 0f 7e c0       	movq   %xmm0,%rax
    86ff:	59                   	pop    %rcx
    8700:	66 48 0f 6e c1       	movq   %rcx,%xmm0
    8705:	66 48 0f 6e c8       	movq   %rax,%xmm1
    870a:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    870e:	66 48 0f 7e c0       	movq   %xmm0,%rax
    8713:	48 89 85 68 fe ff ff 	mov    %rax,-0x198(%rbp)
    871a:	48 8b 85 68 fe ff ff 	mov    -0x198(%rbp),%rax
    8721:	66 48 0f 6e c0       	movq   %rax,%xmm0
    8726:	f2 48 0f 2c c0       	cvttsd2si %xmm0,%rax
    872b:	48 89 85 60 fe ff ff 	mov    %rax,-0x1a0(%rbp)
    8732:	48 8b 85 60 fe ff ff 	mov    -0x1a0(%rbp),%rax
    8739:	50                   	push   %rax
    873a:	48 8b 85 b8 ff ff ff 	mov    -0x48(%rbp),%rax
    8741:	59                   	pop    %rcx
    8742:	48 39 c1             	cmp    %rax,%rcx
    8745:	0f 9d c0             	setge  %al
    8748:	48 0f b6 c0          	movzbq %al,%rax
    874c:	48 85 c0             	test   %rax,%rax
    874f:	0f 84 26 00 00 00    	je     0x877b
    8755:	48 8b 85 a0 fe ff ff 	mov    -0x160(%rbp),%rax
    875c:	50                   	push   %rax
    875d:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
    8764:	00 00 00 
    8767:	59                   	pop    %rcx
    8768:	48 01 c8             	add    %rcx,%rax
    876b:	48 89 85 a0 fe ff ff 	mov    %rax,-0x160(%rbp)
    8772:	31 c0                	xor    %eax,%eax
    8774:	48 89 85 60 fe ff ff 	mov    %rax,-0x1a0(%rbp)
    877b:	48 8b 85 a0 fe ff ff 	mov    -0x160(%rbp),%rax
    8782:	50                   	push   %rax
    8783:	51                   	push   %rcx
    8784:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    878b:	00 00 00 
    878e:	48 89 c1             	mov    %rax,%rcx
    8791:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    8798:	00 00 00 
    879b:	48 89 08             	mov    %rcx,(%rax)
    879e:	48 89 c8             	mov    %rcx,%rax
    87a1:	59                   	pop    %rcx
    87a2:	48 8b bc 24 00 00 00 	mov    0x0(%rsp),%rdi
    87a9:	00 
    87aa:	e8 6b ed ff ff       	call   0x751a
    87af:	48 83 c4 08          	add    $0x8,%rsp
    87b3:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    87ba:	50                   	push   %rax
    87bb:	31 c0                	xor    %eax,%eax
    87bd:	59                   	pop    %rcx
    87be:	48 39 c1             	cmp    %rax,%rcx
    87c1:	0f 9f c0             	setg   %al
    87c4:	48 0f b6 c0          	movzbq %al,%rax
    87c8:	48 85 c0             	test   %rax,%rax
    87cb:	0f 84 90 00 00 00    	je     0x8861
    87d1:	48 b8 2e 00 00 00 00 	movabs $0x2e,%rax
    87d8:	00 00 00 
    87db:	48 83 ec 08          	sub    $0x8,%rsp
    87df:	88 04 24             	mov    %al,(%rsp)
    87e2:	b8 01 00 00 00       	mov    $0x1,%eax
    87e7:	bf 01 00 00 00       	mov    $0x1,%edi
    87ec:	48 89 e6             	mov    %rsp,%rsi
    87ef:	ba 01 00 00 00       	mov    $0x1,%edx
    87f4:	0f 05                	syscall
    87f6:	48 83 c4 08          	add    $0x8,%rsp
    87fa:	31 c0                	xor    %eax,%eax
    87fc:	48 8b 85 60 fe ff ff 	mov    -0x1a0(%rbp),%rax
    8803:	50                   	push   %rax
    8804:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
    880b:	50                   	push   %rax
    880c:	51                   	push   %rcx
    880d:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    8814:	00 00 00 
    8817:	48 89 c1             	mov    %rax,%rcx
    881a:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    8821:	00 00 00 
    8824:	48 89 08             	mov    %rcx,(%rax)
    8827:	48 89 c8             	mov    %rcx,%rax
    882a:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    8831:	00 00 00 
    8834:	48 89 c1             	mov    %rax,%rcx
    8837:	48 b8 a8 00 00 10 00 	movabs $0x100000a8,%rax
    883e:	00 00 00 
    8841:	48 89 08             	mov    %rcx,(%rax)
    8844:	48 89 c8             	mov    %rcx,%rax
    8847:	59                   	pop    %rcx
    8848:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
    884f:	00 
    8850:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
    8857:	00 
    8858:	e8 6a ee ff ff       	call   0x76c7
    885d:	48 83 c4 10          	add    $0x10,%rsp
    8861:	31 c0                	xor    %eax,%eax
    8863:	48 81 c4 b0 01 00 00 	add    $0x1b0,%rsp
    886a:	5d                   	pop    %rbp
    886b:	c3                   	ret


/workspace/.tmp/sig139-census01/reporter.bin:     file format binary


Disassembly of section .data:

000000000588d4be <.data>:
 588d4be:	55                   	push   %rbp
 588d4bf:	48 89 e5             	mov    %rsp,%rbp
 588d4c2:	41 54                	push   %r12
 588d4c4:	48 81 ec d0 0a 00 00 	sub    $0xad0,%rsp
 588d4cb:	48 89 e8             	mov    %rbp,%rax
 588d4ce:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 588d4d4:	48 39 e0             	cmp    %rsp,%rax
 588d4d7:	76 05                	jbe    0x588d4de
 588d4d9:	80 08 00             	orb    $0x0,(%rax)
 588d4dc:	eb f0                	jmp    0x588d4ce
 588d4de:	48 89 f8             	mov    %rdi,%rax
 588d4e1:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 588d4e8:	49 89 fc             	mov    %rdi,%r12
 588d4eb:	48 89 f0             	mov    %rsi,%rax
 588d4ee:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
 588d4f5:	48 89 d0             	mov    %rdx,%rax
 588d4f8:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
 588d4ff:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 588d506:	48 89 c6             	mov    %rax,%rsi
 588d509:	48 8d bd c8 f6 ff ff 	lea    -0x938(%rbp),%rdi
 588d510:	b9 22 01 00 00       	mov    $0x122,%ecx
 588d515:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d518:	48 8d 85 c8 f6 ff ff 	lea    -0x938(%rbp),%rax
 588d51f:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
 588d526:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 588d52d:	48 8b 80 58 00 00 00 	mov    0x58(%rax),%rax
 588d534:	50                   	push   %rax
 588d535:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 588d53c:	00 00 00 
 588d53f:	59                   	pop    %rcx
 588d540:	48 01 c8             	add    %rcx,%rax
 588d543:	50                   	push   %rax
 588d544:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 588d54b:	48 8d 80 58 00 00 00 	lea    0x58(%rax),%rax
 588d552:	5a                   	pop    %rdx
 588d553:	48 89 10             	mov    %rdx,(%rax)
 588d556:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 588d55d:	00 00 00 
 588d560:	50                   	push   %rax
 588d561:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 588d568:	48 8d 80 60 00 00 00 	lea    0x60(%rax),%rax
 588d56f:	5a                   	pop    %rdx
 588d570:	48 89 10             	mov    %rdx,(%rax)
 588d573:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588d57a:	50                   	push   %rax
 588d57b:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 588d582:	48 8b 80 18 00 00 00 	mov    0x18(%rax),%rax
 588d589:	48 89 c6             	mov    %rax,%rsi
 588d58c:	48 8d bd 40 f5 ff ff 	lea    -0xac0(%rbp),%rdi
 588d593:	b9 31 00 00 00       	mov    $0x31,%ecx
 588d598:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d59b:	48 8d 85 40 f5 ff ff 	lea    -0xac0(%rbp),%rax
 588d5a2:	50                   	push   %rax
 588d5a3:	66 90                	xchg   %ax,%ax
 588d5a5:	51                   	push   %rcx
 588d5a6:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d5ad:	00 00 00 
 588d5b0:	48 89 c1             	mov    %rax,%rcx
 588d5b3:	48 b8 20 80 42 5e 00 	movabs $0x5e428020,%rax
 588d5ba:	00 00 00 
 588d5bd:	48 89 08             	mov    %rcx,(%rax)
 588d5c0:	48 89 c8             	mov    %rcx,%rax
 588d5c3:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d5ca:	00 00 00 
 588d5cd:	48 89 c1             	mov    %rax,%rcx
 588d5d0:	48 b8 28 80 42 5e 00 	movabs $0x5e428028,%rax
 588d5d7:	00 00 00 
 588d5da:	48 89 08             	mov    %rcx,(%rax)
 588d5dd:	48 89 c8             	mov    %rcx,%rax
 588d5e0:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d5e7:	00 00 00 
 588d5ea:	48 89 c1             	mov    %rax,%rcx
 588d5ed:	48 b8 e8 80 42 5e 00 	movabs $0x5e4280e8,%rax
 588d5f4:	00 00 00 
 588d5f7:	48 89 08             	mov    %rcx,(%rax)
 588d5fa:	48 89 c8             	mov    %rcx,%rax
 588d5fd:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d604:	00 00 00 
 588d607:	48 89 c1             	mov    %rax,%rcx
 588d60a:	48 b8 68 81 42 5e 00 	movabs $0x5e428168,%rax
 588d611:	00 00 00 
 588d614:	48 89 08             	mov    %rcx,(%rax)
 588d617:	48 89 c8             	mov    %rcx,%rax
 588d61a:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d621:	00 00 00 
 588d624:	48 89 c1             	mov    %rax,%rcx
 588d627:	48 b8 e8 81 42 5e 00 	movabs $0x5e4281e8,%rax
 588d62e:	00 00 00 
 588d631:	48 89 08             	mov    %rcx,(%rax)
 588d634:	48 89 c8             	mov    %rcx,%rax
 588d637:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d63e:	00 00 00 
 588d641:	48 89 c1             	mov    %rax,%rcx
 588d644:	48 b8 68 82 42 5e 00 	movabs $0x5e428268,%rax
 588d64b:	00 00 00 
 588d64e:	48 89 08             	mov    %rcx,(%rax)
 588d651:	48 89 c8             	mov    %rcx,%rax
 588d654:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d65b:	00 00 00 
 588d65e:	48 89 c1             	mov    %rax,%rcx
 588d661:	48 b8 e8 82 42 5e 00 	movabs $0x5e4282e8,%rax
 588d668:	00 00 00 
 588d66b:	48 89 08             	mov    %rcx,(%rax)
 588d66e:	48 89 c8             	mov    %rcx,%rax
 588d671:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d678:	00 00 00 
 588d67b:	48 89 c1             	mov    %rax,%rcx
 588d67e:	48 b8 68 83 42 5e 00 	movabs $0x5e428368,%rax
 588d685:	00 00 00 
 588d688:	48 89 08             	mov    %rcx,(%rax)
 588d68b:	48 89 c8             	mov    %rcx,%rax
 588d68e:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d695:	00 00 00 
 588d698:	48 89 c1             	mov    %rax,%rcx
 588d69b:	48 b8 e8 83 42 5e 00 	movabs $0x5e4283e8,%rax
 588d6a2:	00 00 00 
 588d6a5:	48 89 08             	mov    %rcx,(%rax)
 588d6a8:	48 89 c8             	mov    %rcx,%rax
 588d6ab:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d6b2:	00 00 00 
 588d6b5:	48 89 c1             	mov    %rax,%rcx
 588d6b8:	48 b8 68 84 42 5e 00 	movabs $0x5e428468,%rax
 588d6bf:	00 00 00 
 588d6c2:	48 89 08             	mov    %rcx,(%rax)
 588d6c5:	48 89 c8             	mov    %rcx,%rax
 588d6c8:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d6cf:	00 00 00 
 588d6d2:	48 89 c1             	mov    %rax,%rcx
 588d6d5:	48 b8 e8 84 42 5e 00 	movabs $0x5e4284e8,%rax
 588d6dc:	00 00 00 
 588d6df:	48 89 08             	mov    %rcx,(%rax)
 588d6e2:	48 89 c8             	mov    %rcx,%rax
 588d6e5:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d6ec:	00 00 00 
 588d6ef:	48 89 c1             	mov    %rax,%rcx
 588d6f2:	48 b8 f0 80 42 5e 00 	movabs $0x5e4280f0,%rax
 588d6f9:	00 00 00 
 588d6fc:	48 89 08             	mov    %rcx,(%rax)
 588d6ff:	48 89 c8             	mov    %rcx,%rax
 588d702:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d709:	00 00 00 
 588d70c:	48 89 c1             	mov    %rax,%rcx
 588d70f:	48 b8 70 81 42 5e 00 	movabs $0x5e428170,%rax
 588d716:	00 00 00 
 588d719:	48 89 08             	mov    %rcx,(%rax)
 588d71c:	48 89 c8             	mov    %rcx,%rax
 588d71f:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d726:	00 00 00 
 588d729:	48 89 c1             	mov    %rax,%rcx
 588d72c:	48 b8 f0 81 42 5e 00 	movabs $0x5e4281f0,%rax
 588d733:	00 00 00 
 588d736:	48 89 08             	mov    %rcx,(%rax)
 588d739:	48 89 c8             	mov    %rcx,%rax
 588d73c:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d743:	00 00 00 
 588d746:	48 89 c1             	mov    %rax,%rcx
 588d749:	48 b8 70 82 42 5e 00 	movabs $0x5e428270,%rax
 588d750:	00 00 00 
 588d753:	48 89 08             	mov    %rcx,(%rax)
 588d756:	48 89 c8             	mov    %rcx,%rax
 588d759:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d760:	00 00 00 
 588d763:	48 89 c1             	mov    %rax,%rcx
 588d766:	48 b8 f0 82 42 5e 00 	movabs $0x5e4282f0,%rax
 588d76d:	00 00 00 
 588d770:	48 89 08             	mov    %rcx,(%rax)
 588d773:	48 89 c8             	mov    %rcx,%rax
 588d776:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d77d:	00 00 00 
 588d780:	48 89 c1             	mov    %rax,%rcx
 588d783:	48 b8 70 83 42 5e 00 	movabs $0x5e428370,%rax
 588d78a:	00 00 00 
 588d78d:	48 89 08             	mov    %rcx,(%rax)
 588d790:	48 89 c8             	mov    %rcx,%rax
 588d793:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d79a:	00 00 00 
 588d79d:	48 89 c1             	mov    %rax,%rcx
 588d7a0:	48 b8 f0 83 42 5e 00 	movabs $0x5e4283f0,%rax
 588d7a7:	00 00 00 
 588d7aa:	48 89 08             	mov    %rcx,(%rax)
 588d7ad:	48 89 c8             	mov    %rcx,%rax
 588d7b0:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d7b7:	00 00 00 
 588d7ba:	48 89 c1             	mov    %rax,%rcx
 588d7bd:	48 b8 70 84 42 5e 00 	movabs $0x5e428470,%rax
 588d7c4:	00 00 00 
 588d7c7:	48 89 08             	mov    %rcx,(%rax)
 588d7ca:	48 89 c8             	mov    %rcx,%rax
 588d7cd:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 588d7d4:	00 00 00 
 588d7d7:	48 89 c1             	mov    %rax,%rcx
 588d7da:	48 b8 f0 84 42 5e 00 	movabs $0x5e4284f0,%rax
 588d7e1:	00 00 00 
 588d7e4:	48 89 08             	mov    %rcx,(%rax)
 588d7e7:	48 89 c8             	mov    %rcx,%rax
 588d7ea:	59                   	pop    %rcx
 588d7eb:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
 588d7f2:	00 
 588d7f3:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
 588d7fa:	00 
 588d7fb:	e8 e3 a2 e1 ff       	call   0x56a7ae3
 588d800:	48 83 c4 10          	add    $0x10,%rsp
 588d804:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 588d80b:	48 89 c2             	mov    %rax,%rdx
 588d80e:	4c 89 e1             	mov    %r12,%rcx
 588d811:	48 89 d6             	mov    %rdx,%rsi
 588d814:	48 89 cf             	mov    %rcx,%rdi
 588d817:	b9 22 01 00 00       	mov    $0x122,%ecx
 588d81c:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d81f:	4c 89 e0             	mov    %r12,%rax
 588d822:	48 81 c4 d0 0a 00 00 	add    $0xad0,%rsp
 588d829:	4c 89 e0             	mov    %r12,%rax
 588d82c:	41 5c                	pop    %r12
 588d82e:	5d                   	pop    %rbp
 588d82f:	c3                   	ret
 588d830:	55                   	push   %rbp
 588d831:	48 89 e5             	mov    %rsp,%rbp
 588d834:	41 54                	push   %r12
 588d836:	48 81 ec 60 1b 00 00 	sub    $0x1b60,%rsp
 588d83d:	48 89 e8             	mov    %rbp,%rax
 588d840:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 588d846:	48 39 e0             	cmp    %rsp,%rax
 588d849:	76 05                	jbe    0x588d850
 588d84b:	80 08 00             	orb    $0x0,(%rax)
 588d84e:	eb f0                	jmp    0x588d840
 588d850:	48 89 f8             	mov    %rdi,%rax
 588d853:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 588d85a:	49 89 fc             	mov    %rdi,%r12
 588d85d:	48 89 f0             	mov    %rsi,%rax
 588d860:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
 588d867:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 588d86e:	48 89 c6             	mov    %rax,%rsi
 588d871:	48 8d bd d8 f6 ff ff 	lea    -0x928(%rbp),%rdi
 588d878:	b9 22 01 00 00       	mov    $0x122,%ecx
 588d87d:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d880:	48 8d 85 d8 f6 ff ff 	lea    -0x928(%rbp),%rax
 588d887:	50                   	push   %rax
 588d888:	48 b8 92 19 00 00 00 	movabs $0x1992,%rax
 588d88f:	00 00 00 
 588d892:	50                   	push   %rax
 588d893:	48 8b b4 24 08 00 00 	mov    0x8(%rsp),%rsi
 588d89a:	00 
 588d89b:	48 8b 94 24 00 00 00 	mov    0x0(%rsp),%rdx
 588d8a2:	00 
 588d8a3:	48 8d 85 c8 ed ff ff 	lea    -0x1238(%rbp),%rax
 588d8aa:	48 89 c7             	mov    %rax,%rdi
 588d8ad:	e8 0c fc ff ff       	call   0x588d4be
 588d8b2:	48 83 c4 10          	add    $0x10,%rsp
 588d8b6:	48 89 c6             	mov    %rax,%rsi
 588d8b9:	48 8d bd b0 e4 ff ff 	lea    -0x1b50(%rbp),%rdi
 588d8c0:	b9 22 01 00 00       	mov    $0x122,%ecx
 588d8c5:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d8c8:	48 8d 85 b0 e4 ff ff 	lea    -0x1b50(%rbp),%rax
 588d8cf:	48 89 85 c0 ed ff ff 	mov    %rax,-0x1240(%rbp)
 588d8d6:	66 90                	xchg   %ax,%ax
 588d8d8:	51                   	push   %rcx
 588d8d9:	59                   	pop    %rcx
 588d8da:	e8 9d d0 e7 ff       	call   0x570a97c
 588d8df:	50                   	push   %rax
 588d8e0:	48 8b 85 c0 ed ff ff 	mov    -0x1240(%rbp),%rax
 588d8e7:	5a                   	pop    %rdx
 588d8e8:	48 89 10             	mov    %rdx,(%rax)
 588d8eb:	48 b8 ff ff ff ff ff 	movabs $0xffffffffffffffff,%rax
 588d8f2:	ff ff ff 
 588d8f5:	50                   	push   %rax
 588d8f6:	48 8b 85 c0 ed ff ff 	mov    -0x1240(%rbp),%rax
 588d8fd:	48 8d 80 10 00 00 00 	lea    0x10(%rax),%rax
 588d904:	5a                   	pop    %rdx
 588d905:	48 89 10             	mov    %rdx,(%rax)
 588d908:	31 c0                	xor    %eax,%eax
 588d90a:	50                   	push   %rax
 588d90b:	48 8b 85 c0 ed ff ff 	mov    -0x1240(%rbp),%rax
 588d912:	48 8d 80 20 00 00 00 	lea    0x20(%rax),%rax
 588d919:	5a                   	pop    %rdx
 588d91a:	48 89 10             	mov    %rdx,(%rax)
 588d91d:	48 8b 85 c0 ed ff ff 	mov    -0x1240(%rbp),%rax
 588d924:	48 89 c2             	mov    %rax,%rdx
 588d927:	4c 89 e1             	mov    %r12,%rcx
 588d92a:	48 89 d6             	mov    %rdx,%rsi
 588d92d:	48 89 cf             	mov    %rcx,%rdi
 588d930:	b9 22 01 00 00       	mov    $0x122,%ecx
 588d935:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d938:	4c 89 e0             	mov    %r12,%rax
 588d93b:	48 81 c4 60 1b 00 00 	add    $0x1b60,%rsp
 588d942:	4c 89 e0             	mov    %r12,%rax
 588d945:	41 5c                	pop    %r12
 588d947:	5d                   	pop    %rbp
 588d948:	c3                   	ret
 588d949:	55                   	push   %rbp
 588d94a:	48 89 e5             	mov    %rsp,%rbp
 588d94d:	41 54                	push   %r12
 588d94f:	48 81 ec 40 09 00 00 	sub    $0x940,%rsp
 588d956:	48 89 e8             	mov    %rbp,%rax
 588d959:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 588d95f:	48 39 e0             	cmp    %rsp,%rax
 588d962:	76 05                	jbe    0x588d969
 588d964:	80 08 00             	orb    $0x0,(%rax)
 588d967:	eb f0                	jmp    0x588d959
 588d969:	48 89 f8             	mov    %rdi,%rax
 588d96c:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 588d973:	49 89 fc             	mov    %rdi,%r12
 588d976:	48 89 f0             	mov    %rsi,%rax
 588d979:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
 588d980:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 588d987:	48 89 c6             	mov    %rax,%rsi
 588d98a:	48 8d bd d0 f6 ff ff 	lea    -0x930(%rbp),%rdi
 588d991:	b9 22 01 00 00       	mov    $0x122,%ecx
 588d996:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d999:	48 8d 85 d0 f6 ff ff 	lea    -0x930(%rbp),%rax
 588d9a0:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
 588d9a7:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588d9ae:	48 8b 80 40 00 00 00 	mov    0x40(%rax),%rax
 588d9b5:	50                   	push   %rax
 588d9b6:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 588d9bd:	00 00 00 
 588d9c0:	59                   	pop    %rcx
 588d9c1:	48 01 c8             	add    %rcx,%rax
 588d9c4:	50                   	push   %rax
 588d9c5:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588d9cc:	48 8d 80 40 00 00 00 	lea    0x40(%rax),%rax
 588d9d3:	5a                   	pop    %rdx
 588d9d4:	48 89 10             	mov    %rdx,(%rax)
 588d9d7:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588d9de:	48 89 c2             	mov    %rax,%rdx
 588d9e1:	4c 89 e1             	mov    %r12,%rcx
 588d9e4:	48 89 d6             	mov    %rdx,%rsi
 588d9e7:	48 89 cf             	mov    %rcx,%rdi
 588d9ea:	b9 22 01 00 00       	mov    $0x122,%ecx
 588d9ef:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588d9f2:	4c 89 e0             	mov    %r12,%rax
 588d9f5:	48 81 c4 40 09 00 00 	add    $0x940,%rsp
 588d9fc:	4c 89 e0             	mov    %r12,%rax
 588d9ff:	41 5c                	pop    %r12
 588da01:	5d                   	pop    %rbp
 588da02:	c3                   	ret
 588da03:	55                   	push   %rbp
 588da04:	48 89 e5             	mov    %rsp,%rbp
 588da07:	41 54                	push   %r12
 588da09:	48 81 ec 40 09 00 00 	sub    $0x940,%rsp
 588da10:	48 89 e8             	mov    %rbp,%rax
 588da13:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 588da19:	48 39 e0             	cmp    %rsp,%rax
 588da1c:	76 05                	jbe    0x588da23
 588da1e:	80 08 00             	orb    $0x0,(%rax)
 588da21:	eb f0                	jmp    0x588da13
 588da23:	48 89 f8             	mov    %rdi,%rax
 588da26:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 588da2d:	49 89 fc             	mov    %rdi,%r12
 588da30:	48 89 f0             	mov    %rsi,%rax
 588da33:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
 588da3a:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 588da41:	48 89 c6             	mov    %rax,%rsi
 588da44:	48 8d bd d0 f6 ff ff 	lea    -0x930(%rbp),%rdi
 588da4b:	b9 22 01 00 00       	mov    $0x122,%ecx
 588da50:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588da53:	48 8d 85 d0 f6 ff ff 	lea    -0x930(%rbp),%rax
 588da5a:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
 588da61:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588da68:	48 8b 80 40 00 00 00 	mov    0x40(%rax),%rax
 588da6f:	50                   	push   %rax
 588da70:	31 c0                	xor    %eax,%eax
 588da72:	59                   	pop    %rcx
 588da73:	48 39 c1             	cmp    %rax,%rcx
 588da76:	0f 9f c0             	setg   %al
 588da79:	48 0f b6 c0          	movzbq %al,%rax
 588da7d:	48 85 c0             	test   %rax,%rax
 588da80:	0f 84 33 00 00 00    	je     0x588dab9
 588da86:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588da8d:	48 8b 80 40 00 00 00 	mov    0x40(%rax),%rax
 588da94:	50                   	push   %rax
 588da95:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 588da9c:	00 00 00 
 588da9f:	59                   	pop    %rcx
 588daa0:	48 87 c1             	xchg   %rax,%rcx
 588daa3:	48 29 c8             	sub    %rcx,%rax
 588daa6:	50                   	push   %rax
 588daa7:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588daae:	48 8d 80 40 00 00 00 	lea    0x40(%rax),%rax
 588dab5:	5a                   	pop    %rdx
 588dab6:	48 89 10             	mov    %rdx,(%rax)
 588dab9:	90                   	nop
 588daba:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588dac1:	48 89 c2             	mov    %rax,%rdx
 588dac4:	4c 89 e1             	mov    %r12,%rcx
 588dac7:	48 89 d6             	mov    %rdx,%rsi
 588daca:	48 89 cf             	mov    %rcx,%rdi
 588dacd:	b9 22 01 00 00       	mov    $0x122,%ecx
 588dad2:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588dad5:	4c 89 e0             	mov    %r12,%rax
 588dad8:	48 81 c4 40 09 00 00 	add    $0x940,%rsp
 588dadf:	4c 89 e0             	mov    %r12,%rax
 588dae2:	41 5c                	pop    %r12
 588dae4:	5d                   	pop    %rbp
 588dae5:	c3                   	ret
 588dae6:	55                   	push   %rbp
 588dae7:	48 89 e5             	mov    %rsp,%rbp
 588daea:	41 54                	push   %r12
 588daec:	48 81 ec 30 1d 00 00 	sub    $0x1d30,%rsp
 588daf3:	48 89 e8             	mov    %rbp,%rax
 588daf6:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 588dafc:	48 39 e0             	cmp    %rsp,%rax
 588daff:	76 05                	jbe    0x588db06
 588db01:	80 08 00             	orb    $0x0,(%rax)
 588db04:	eb f0                	jmp    0x588daf6
 588db06:	48 89 f8             	mov    %rdi,%rax
 588db09:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 588db10:	49 89 fc             	mov    %rdi,%r12
 588db13:	48 89 f0             	mov    %rsi,%rax
 588db16:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
 588db1d:	48 89 d0             	mov    %rdx,%rax
 588db20:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
 588db27:	48 89 c8             	mov    %rcx,%rax
 588db2a:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
 588db31:	4c 89 c0             	mov    %r8,%rax
 588db34:	48 89 85 d0 ff ff ff 	mov    %rax,-0x30(%rbp)
 588db3b:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 588db42:	48 89 c6             	mov    %rax,%rsi
 588db45:	48 8d bd b8 f6 ff ff 	lea    -0x948(%rbp),%rdi
 588db4c:	b9 22 01 00 00       	mov    $0x122,%ecx
 588db51:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588db54:	48 8d 85 b8 f6 ff ff 	lea    -0x948(%rbp),%rax
 588db5b:	48 89 85 c8 ff ff ff 	mov    %rax,-0x38(%rbp)
 588db62:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
 588db69:	48 89 85 b0 f6 ff ff 	mov    %rax,-0x950(%rbp)
 588db70:	48 8b 85 c8 ff ff ff 	mov    -0x38(%rbp),%rax
 588db77:	48 8b 80 38 00 00 00 	mov    0x38(%rax),%rax
 588db7e:	48 89 85 a8 f6 ff ff 	mov    %rax,-0x958(%rbp)
 588db85:	48 8b 85 a8 f6 ff ff 	mov    -0x958(%rbp),%rax
 588db8c:	48 8b 80 f0 00 00 00 	mov    0xf0(%rax),%rax
 588db93:	50                   	push   %rax
 588db94:	48 b8 00 10 00 00 00 	movabs $0x1000,%rax
 588db9b:	00 00 00 
 588db9e:	59                   	pop    %rcx
 588db9f:	48 39 c1             	cmp    %rax,%rcx
 588dba2:	0f 9c c0             	setl   %al
 588dba5:	48 0f b6 c0          	movzbq %al,%rax
 588dba9:	48 85 c0             	test   %rax,%rax
 588dbac:	0f 84 30 05 00 00    	je     0x588e0e2
 588dbb2:	48 8b 85 a8 f6 ff ff 	mov    -0x958(%rbp),%rax
 588dbb9:	48 8b 80 f0 00 00 00 	mov    0xf0(%rax),%rax
 588dbc0:	48 89 85 a0 f6 ff ff 	mov    %rax,-0x960(%rbp)
 588dbc7:	48 8b 85 a8 f6 ff ff 	mov    -0x958(%rbp),%rax
 588dbce:	48 8b 00             	mov    (%rax),%rax
 588dbd1:	50                   	push   %rax
 588dbd2:	48 8b 85 a0 f6 ff ff 	mov    -0x960(%rbp),%rax
 588dbd9:	50                   	push   %rax
 588dbda:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 588dbe1:	48 89 85 98 f6 ff ff 	mov    %rax,-0x968(%rbp)
 588dbe8:	48 b8 88 01 00 00 00 	movabs $0x188,%rax
 588dbef:	00 00 00 
 588dbf2:	e8 3e 45 b7 fa       	call   0x402135
 588dbf7:	48 89 85 90 f6 ff ff 	mov    %rax,-0x970(%rbp)
 588dbfe:	48 8b 85 98 f6 ff ff 	mov    -0x968(%rbp),%rax
 588dc05:	48 89 c2             	mov    %rax,%rdx
 588dc08:	48 8b 85 90 f6 ff ff 	mov    -0x970(%rbp),%rax
 588dc0f:	48 89 d6             	mov    %rdx,%rsi
 588dc12:	48 89 c7             	mov    %rax,%rdi
 588dc15:	b9 31 00 00 00       	mov    $0x31,%ecx
 588dc1a:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 588dc1d:	48 89 c2             	mov    %rax,%rdx
 588dc20:	59                   	pop    %rcx
 588dc21:	58                   	pop    %rax
 588dc22:	48 89 14 c8          	mov    %rdx,(%rax,%rcx,8)
 588dc26:	48 8b 85 a8 f6 ff ff 	mov    -0x958(%rbp),%rax
 588dc2d:	48 8b 80 08 00 00 00 	mov    0x8(%rax),%rax
 588dc34:	50                   	push   %rax
 588dc35:	48 8b 85 a0 f6 ff ff 	mov    -0x960(%rbp),%rax
 588dc3c:	50                   	push   %rax
 588dc3d:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 588dc44:	48 89 c2             	mov    %rax,%rdx
 588dc47:	59                   	pop    %rcx
 588dc48:	58                   	pop    %rax
 588dc49:	48 89 14 c8          	mov    %rdx,(%rax,%rcx,8)
 588dc4d:	48 8b 85 a8 f6 ff ff 	mov    -0x958(%rbp),%rax
 588dc54:	48 8b 80 10 00 00 00 	mov    0x10(%rax),%rax
 588dc5b:	50                   	push   %rax
 588dc5c:	48 8b 85 a0 f6 ff ff 	mov    -0x960(%rbp),%rax
 588dc63:	50                   	push   %rax
 588dc64:	48 8b 85 c8 ff ff ff 	mov    -0x38(%rbp),%rax
 588dc6b:	48 8b 80 40 00 00 00 	mov    0x40(%rax),%rax
 588dc72:	48 89 c2             	mov    %rax,%rdx
 588dc75:	59                   	pop    %rcx
 588dc76:	58                   	pop    %rax
 588dc77:	48 89 14 c8          	mov    %rdx,(%rax,%rcx,8)
 588dc7b:	48 8b 85 a8 f6 ff ff 	mov    -0x958(%rbp),%rax
 588dc82:	48 8b 80 18 00 00 00 	mov    0x18(%rax),%rax
 588dc89:	50                   	push   %rax
 588dc8a:	48 8b 85 a0 f6 ff ff 	mov    -0x960(%rbp),%rax
 588dc91:	50                   	push   %rax
 588dc92:	51                   	push   %rcx
 588dc93:	59                   	pop    %rcx
 588dc94:	48 8d 85 08 f5 ff ff 	lea    -0xaf8(%rbp),%rax
 588dc9b:	48 89 c7             	mov    %rax,%rdi
 588dc9e:	e8 cd a2 43 fc       	call   0x1cc7f70
 588dca3:	48 89 85 00 f5 ff ff 	mov    %rax,-0xb00(%rbp)
 588dcaa:	48 b8 88 01 00 00 00 	movabs $0x188,%rax
 588dcb1:	00 00 00 
 588dcb4:	e8 7c 44 b7 fa       	call   0x402135
 588dcb9:	48                   	rex.W
 588dcba:	89                   	.byte 0x89
 588dcbb:	85 f8                	test   %edi,%eax
 588dcbd:	f4                   	hlt

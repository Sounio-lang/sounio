
/workspace/.tmp/sig139-census01/family_b.bin:     file format binary


Disassembly of section .data:

00000000059e5163 <.data>:
 59e5163:	55                   	push   %rbp
 59e5164:	48 89 e5             	mov    %rsp,%rbp
 59e5167:	48 81 ec 80 09 00 00 	sub    $0x980,%rsp
 59e516e:	48 89 e8             	mov    %rbp,%rax
 59e5171:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 59e5177:	48 39 e0             	cmp    %rsp,%rax
 59e517a:	76 05                	jbe    0x59e5181
 59e517c:	80 08 00             	orb    $0x0,(%rax)
 59e517f:	eb f0                	jmp    0x59e5171
 59e5181:	48 89 f8             	mov    %rdi,%rax
 59e5184:	48 89 85 f8 ff ff ff 	mov    %rax,-0x8(%rbp)
 59e518b:	48 89 f0             	mov    %rsi,%rax
 59e518e:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 59e5195:	48 89 d0             	mov    %rdx,%rax
 59e5198:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
 59e519f:	31 c0                	xor    %eax,%eax
 59e51a1:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
 59e51a8:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e51af:	00 00 00 
 59e51b2:	48 f7 d8             	neg    %rax
 59e51b5:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
 59e51bc:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 59e51c3:	50                   	push   %rax
 59e51c4:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
 59e51cb:	48 8b 80 68 00 00 00 	mov    0x68(%rax),%rax
 59e51d2:	48 8b 80 00 80 00 00 	mov    0x8000(%rax),%rax
 59e51d9:	59                   	pop    %rcx
 59e51da:	48 39 c1             	cmp    %rax,%rcx
 59e51dd:	0f 9c c0             	setl   %al
 59e51e0:	48 0f b6 c0          	movzbq %al,%rax
 59e51e4:	48 85 c0             	test   %rax,%rax
 59e51e7:	0f 84 3e 06 00 00    	je     0x59e582b
 59e51ed:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
 59e51f4:	48 8b 80 68 00 00 00 	mov    0x68(%rax),%rax
 59e51fb:	50                   	push   %rax
 59e51fc:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 59e5203:	48 89 c2             	mov    %rax,%rdx
 59e5206:	59                   	pop    %rcx
 59e5207:	48 8b 04 d1          	mov    (%rcx,%rdx,8),%rax
 59e520b:	48 89 c6             	mov    %rax,%rsi
 59e520e:	48 8d bd a8 fc ff ff 	lea    -0x358(%rbp),%rdi
 59e5215:	b9 65 00 00 00       	mov    $0x65,%ecx
 59e521a:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e521d:	48 8d 85 a8 fc ff ff 	lea    -0x358(%rbp),%rax
 59e5224:	48 89 85 d0 ff ff ff 	mov    %rax,-0x30(%rbp)
 59e522b:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
 59e5232:	48 8d 80 88 01 00 00 	lea    0x188(%rax),%rax
 59e5239:	48 89 c6             	mov    %rax,%rsi
 59e523c:	48 8d bd 20 fb ff ff 	lea    -0x4e0(%rbp),%rdi
 59e5243:	b9 31 00 00 00       	mov    $0x31,%ecx
 59e5248:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e524b:	48 8d 85 20 fb ff ff 	lea    -0x4e0(%rbp),%rax
 59e5252:	50                   	push   %rax
 59e5253:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 59e525a:	48 89 c6             	mov    %rax,%rsi
 59e525d:	48 8d bd 98 f9 ff ff 	lea    -0x668(%rbp),%rdi
 59e5264:	b9 31 00 00 00       	mov    $0x31,%ecx
 59e5269:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e526c:	48 8d 85 98 f9 ff ff 	lea    -0x668(%rbp),%rax
 59e5273:	50                   	push   %rax
 59e5274:	66 90                	xchg   %ax,%ax
 59e5276:	51                   	push   %rcx
 59e5277:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e527e:	00 00 00 
 59e5281:	48 89 c1             	mov    %rax,%rcx
 59e5284:	48 b8 20 80 42 5e 00 	movabs $0x5e428020,%rax
 59e528b:	00 00 00 
 59e528e:	48 89 08             	mov    %rcx,(%rax)
 59e5291:	48 89 c8             	mov    %rcx,%rax
 59e5294:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e529b:	00 00 00 
 59e529e:	48 89 c1             	mov    %rax,%rcx
 59e52a1:	48 b8 28 80 42 5e 00 	movabs $0x5e428028,%rax
 59e52a8:	00 00 00 
 59e52ab:	48 89 08             	mov    %rcx,(%rax)
 59e52ae:	48 89 c8             	mov    %rcx,%rax
 59e52b1:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e52b8:	00 00 00 
 59e52bb:	48 89 c1             	mov    %rax,%rcx
 59e52be:	48 b8 e8 80 42 5e 00 	movabs $0x5e4280e8,%rax
 59e52c5:	00 00 00 
 59e52c8:	48 89 08             	mov    %rcx,(%rax)
 59e52cb:	48 89 c8             	mov    %rcx,%rax
 59e52ce:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e52d5:	00 00 00 
 59e52d8:	48 89 c1             	mov    %rax,%rcx
 59e52db:	48 b8 68 81 42 5e 00 	movabs $0x5e428168,%rax
 59e52e2:	00 00 00 
 59e52e5:	48 89 08             	mov    %rcx,(%rax)
 59e52e8:	48 89 c8             	mov    %rcx,%rax
 59e52eb:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e52f2:	00 00 00 
 59e52f5:	48 89 c1             	mov    %rax,%rcx
 59e52f8:	48 b8 e8 81 42 5e 00 	movabs $0x5e4281e8,%rax
 59e52ff:	00 00 00 
 59e5302:	48 89 08             	mov    %rcx,(%rax)
 59e5305:	48 89 c8             	mov    %rcx,%rax
 59e5308:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e530f:	00 00 00 
 59e5312:	48 89 c1             	mov    %rax,%rcx
 59e5315:	48 b8 68 82 42 5e 00 	movabs $0x5e428268,%rax
 59e531c:	00 00 00 
 59e531f:	48 89 08             	mov    %rcx,(%rax)
 59e5322:	48 89 c8             	mov    %rcx,%rax
 59e5325:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e532c:	00 00 00 
 59e532f:	48 89 c1             	mov    %rax,%rcx
 59e5332:	48 b8 e8 82 42 5e 00 	movabs $0x5e4282e8,%rax
 59e5339:	00 00 00 
 59e533c:	48 89 08             	mov    %rcx,(%rax)
 59e533f:	48 89 c8             	mov    %rcx,%rax
 59e5342:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5349:	00 00 00 
 59e534c:	48 89 c1             	mov    %rax,%rcx
 59e534f:	48 b8 68 83 42 5e 00 	movabs $0x5e428368,%rax
 59e5356:	00 00 00 
 59e5359:	48 89 08             	mov    %rcx,(%rax)
 59e535c:	48 89 c8             	mov    %rcx,%rax
 59e535f:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5366:	00 00 00 
 59e5369:	48 89 c1             	mov    %rax,%rcx
 59e536c:	48 b8 e8 83 42 5e 00 	movabs $0x5e4283e8,%rax
 59e5373:	00 00 00 
 59e5376:	48 89 08             	mov    %rcx,(%rax)
 59e5379:	48 89 c8             	mov    %rcx,%rax
 59e537c:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5383:	00 00 00 
 59e5386:	48 89 c1             	mov    %rax,%rcx
 59e5389:	48 b8 68 84 42 5e 00 	movabs $0x5e428468,%rax
 59e5390:	00 00 00 
 59e5393:	48 89 08             	mov    %rcx,(%rax)
 59e5396:	48 89 c8             	mov    %rcx,%rax
 59e5399:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e53a0:	00 00 00 
 59e53a3:	48 89 c1             	mov    %rax,%rcx
 59e53a6:	48 b8 e8 84 42 5e 00 	movabs $0x5e4284e8,%rax
 59e53ad:	00 00 00 
 59e53b0:	48 89 08             	mov    %rcx,(%rax)
 59e53b3:	48 89 c8             	mov    %rcx,%rax
 59e53b6:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e53bd:	00 00 00 
 59e53c0:	48 89 c1             	mov    %rax,%rcx
 59e53c3:	48 b8 f0 80 42 5e 00 	movabs $0x5e4280f0,%rax
 59e53ca:	00 00 00 
 59e53cd:	48 89 08             	mov    %rcx,(%rax)
 59e53d0:	48 89 c8             	mov    %rcx,%rax
 59e53d3:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e53da:	00 00 00 
 59e53dd:	48 89 c1             	mov    %rax,%rcx
 59e53e0:	48 b8 70 81 42 5e 00 	movabs $0x5e428170,%rax
 59e53e7:	00 00 00 
 59e53ea:	48 89 08             	mov    %rcx,(%rax)
 59e53ed:	48 89 c8             	mov    %rcx,%rax
 59e53f0:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e53f7:	00 00 00 
 59e53fa:	48 89 c1             	mov    %rax,%rcx
 59e53fd:	48 b8 f0 81 42 5e 00 	movabs $0x5e4281f0,%rax
 59e5404:	00 00 00 
 59e5407:	48 89 08             	mov    %rcx,(%rax)
 59e540a:	48 89 c8             	mov    %rcx,%rax
 59e540d:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5414:	00 00 00 
 59e5417:	48 89 c1             	mov    %rax,%rcx
 59e541a:	48 b8 70 82 42 5e 00 	movabs $0x5e428270,%rax
 59e5421:	00 00 00 
 59e5424:	48 89 08             	mov    %rcx,(%rax)
 59e5427:	48 89 c8             	mov    %rcx,%rax
 59e542a:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5431:	00 00 00 
 59e5434:	48 89 c1             	mov    %rax,%rcx
 59e5437:	48 b8 f0 82 42 5e 00 	movabs $0x5e4282f0,%rax
 59e543e:	00 00 00 
 59e5441:	48 89 08             	mov    %rcx,(%rax)
 59e5444:	48 89 c8             	mov    %rcx,%rax
 59e5447:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e544e:	00 00 00 
 59e5451:	48 89 c1             	mov    %rax,%rcx
 59e5454:	48 b8 70 83 42 5e 00 	movabs $0x5e428370,%rax
 59e545b:	00 00 00 
 59e545e:	48 89 08             	mov    %rcx,(%rax)
 59e5461:	48 89 c8             	mov    %rcx,%rax
 59e5464:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e546b:	00 00 00 
 59e546e:	48 89 c1             	mov    %rax,%rcx
 59e5471:	48 b8 f0 83 42 5e 00 	movabs $0x5e4283f0,%rax
 59e5478:	00 00 00 
 59e547b:	48 89 08             	mov    %rcx,(%rax)
 59e547e:	48 89 c8             	mov    %rcx,%rax
 59e5481:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5488:	00 00 00 
 59e548b:	48 89 c1             	mov    %rax,%rcx
 59e548e:	48 b8 70 84 42 5e 00 	movabs $0x5e428470,%rax
 59e5495:	00 00 00 
 59e5498:	48 89 08             	mov    %rcx,(%rax)
 59e549b:	48 89 c8             	mov    %rcx,%rax
 59e549e:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e54a5:	00 00 00 
 59e54a8:	48 89 c1             	mov    %rax,%rcx
 59e54ab:	48 b8 f0 84 42 5e 00 	movabs $0x5e4284f0,%rax
 59e54b2:	00 00 00 
 59e54b5:	48 89 08             	mov    %rcx,(%rax)
 59e54b8:	48 89 c8             	mov    %rcx,%rax
 59e54bb:	59                   	pop    %rcx
 59e54bc:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
 59e54c3:	00 
 59e54c4:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
 59e54cb:	00 
 59e54cc:	e8 20 41 2e fc       	call   0x1cc95f1
 59e54d1:	48 83 c4 10          	add    $0x10,%rsp
 59e54d5:	48 85 c0             	test   %rax,%rax
 59e54d8:	0f 84 2a 03 00 00    	je     0x59e5808
 59e54de:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
 59e54e5:	48 8b 80 80 01 00 00 	mov    0x180(%rax),%rax
 59e54ec:	50                   	push   %rax
 59e54ed:	31 c0                	xor    %eax,%eax
 59e54ef:	59                   	pop    %rcx
 59e54f0:	48 39 c1             	cmp    %rax,%rcx
 59e54f3:	0f 9f c0             	setg   %al
 59e54f6:	48 0f b6 c0          	movzbq %al,%rax
 59e54fa:	48 85 c0             	test   %rax,%rax
 59e54fd:	0f 84 c2 02 00 00    	je     0x59e57c5
 59e5503:	48 8b 85 d0 ff ff ff 	mov    -0x30(%rbp),%rax
 59e550a:	48 89 c6             	mov    %rax,%rsi
 59e550d:	48 8d bd 10 f8 ff ff 	lea    -0x7f0(%rbp),%rdi
 59e5514:	b9 31 00 00 00       	mov    $0x31,%ecx
 59e5519:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e551c:	48 8d 85 10 f8 ff ff 	lea    -0x7f0(%rbp),%rax
 59e5523:	50                   	push   %rax
 59e5524:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
 59e552b:	48 89 c6             	mov    %rax,%rsi
 59e552e:	48 8d bd 88 f6 ff ff 	lea    -0x978(%rbp),%rdi
 59e5535:	b9 31 00 00 00       	mov    $0x31,%ecx
 59e553a:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e553d:	48 8d 85 88 f6 ff ff 	lea    -0x978(%rbp),%rax
 59e5544:	50                   	push   %rax
 59e5545:	66 90                	xchg   %ax,%ax
 59e5547:	51                   	push   %rcx
 59e5548:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e554f:	00 00 00 
 59e5552:	48 89 c1             	mov    %rax,%rcx
 59e5555:	48 b8 20 80 42 5e 00 	movabs $0x5e428020,%rax
 59e555c:	00 00 00 
 59e555f:	48 89 08             	mov    %rcx,(%rax)
 59e5562:	48 89 c8             	mov    %rcx,%rax
 59e5565:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e556c:	00 00 00 
 59e556f:	48 89 c1             	mov    %rax,%rcx
 59e5572:	48 b8 28 80 42 5e 00 	movabs $0x5e428028,%rax
 59e5579:	00 00 00 
 59e557c:	48 89 08             	mov    %rcx,(%rax)
 59e557f:	48 89 c8             	mov    %rcx,%rax
 59e5582:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5589:	00 00 00 
 59e558c:	48 89 c1             	mov    %rax,%rcx
 59e558f:	48 b8 e8 80 42 5e 00 	movabs $0x5e4280e8,%rax
 59e5596:	00 00 00 
 59e5599:	48 89 08             	mov    %rcx,(%rax)
 59e559c:	48 89 c8             	mov    %rcx,%rax
 59e559f:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e55a6:	00 00 00 
 59e55a9:	48 89 c1             	mov    %rax,%rcx
 59e55ac:	48 b8 68 81 42 5e 00 	movabs $0x5e428168,%rax
 59e55b3:	00 00 00 
 59e55b6:	48 89 08             	mov    %rcx,(%rax)
 59e55b9:	48 89 c8             	mov    %rcx,%rax
 59e55bc:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e55c3:	00 00 00 
 59e55c6:	48 89 c1             	mov    %rax,%rcx
 59e55c9:	48 b8 e8 81 42 5e 00 	movabs $0x5e4281e8,%rax
 59e55d0:	00 00 00 
 59e55d3:	48 89 08             	mov    %rcx,(%rax)
 59e55d6:	48 89 c8             	mov    %rcx,%rax
 59e55d9:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e55e0:	00 00 00 
 59e55e3:	48 89 c1             	mov    %rax,%rcx
 59e55e6:	48 b8 68 82 42 5e 00 	movabs $0x5e428268,%rax
 59e55ed:	00 00 00 
 59e55f0:	48 89 08             	mov    %rcx,(%rax)
 59e55f3:	48 89 c8             	mov    %rcx,%rax
 59e55f6:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e55fd:	00 00 00 
 59e5600:	48 89 c1             	mov    %rax,%rcx
 59e5603:	48 b8 e8 82 42 5e 00 	movabs $0x5e4282e8,%rax
 59e560a:	00 00 00 
 59e560d:	48 89 08             	mov    %rcx,(%rax)
 59e5610:	48 89 c8             	mov    %rcx,%rax
 59e5613:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e561a:	00 00 00 
 59e561d:	48 89 c1             	mov    %rax,%rcx
 59e5620:	48 b8 68 83 42 5e 00 	movabs $0x5e428368,%rax
 59e5627:	00 00 00 
 59e562a:	48 89 08             	mov    %rcx,(%rax)
 59e562d:	48 89 c8             	mov    %rcx,%rax
 59e5630:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5637:	00 00 00 
 59e563a:	48 89 c1             	mov    %rax,%rcx
 59e563d:	48 b8 e8 83 42 5e 00 	movabs $0x5e4283e8,%rax
 59e5644:	00 00 00 
 59e5647:	48 89 08             	mov    %rcx,(%rax)
 59e564a:	48 89 c8             	mov    %rcx,%rax
 59e564d:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5654:	00 00 00 
 59e5657:	48 89 c1             	mov    %rax,%rcx
 59e565a:	48 b8 68 84 42 5e 00 	movabs $0x5e428468,%rax
 59e5661:	00 00 00 
 59e5664:	48 89 08             	mov    %rcx,(%rax)
 59e5667:	48 89 c8             	mov    %rcx,%rax
 59e566a:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5671:	00 00 00 
 59e5674:	48 89 c1             	mov    %rax,%rcx
 59e5677:	48 b8 e8 84 42 5e 00 	movabs $0x5e4284e8,%rax
 59e567e:	00 00 00 
 59e5681:	48 89 08             	mov    %rcx,(%rax)
 59e5684:	48 89 c8             	mov    %rcx,%rax
 59e5687:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e568e:	00 00 00 
 59e5691:	48 89 c1             	mov    %rax,%rcx
 59e5694:	48 b8 f0 80 42 5e 00 	movabs $0x5e4280f0,%rax
 59e569b:	00 00 00 
 59e569e:	48 89 08             	mov    %rcx,(%rax)
 59e56a1:	48 89 c8             	mov    %rcx,%rax
 59e56a4:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e56ab:	00 00 00 
 59e56ae:	48 89 c1             	mov    %rax,%rcx
 59e56b1:	48 b8 70 81 42 5e 00 	movabs $0x5e428170,%rax
 59e56b8:	00 00 00 
 59e56bb:	48 89 08             	mov    %rcx,(%rax)
 59e56be:	48 89 c8             	mov    %rcx,%rax
 59e56c1:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e56c8:	00 00 00 
 59e56cb:	48 89 c1             	mov    %rax,%rcx
 59e56ce:	48 b8 f0 81 42 5e 00 	movabs $0x5e4281f0,%rax
 59e56d5:	00 00 00 
 59e56d8:	48 89 08             	mov    %rcx,(%rax)
 59e56db:	48 89 c8             	mov    %rcx,%rax
 59e56de:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e56e5:	00 00 00 
 59e56e8:	48 89 c1             	mov    %rax,%rcx
 59e56eb:	48 b8 70 82 42 5e 00 	movabs $0x5e428270,%rax
 59e56f2:	00 00 00 
 59e56f5:	48 89 08             	mov    %rcx,(%rax)
 59e56f8:	48 89 c8             	mov    %rcx,%rax
 59e56fb:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5702:	00 00 00 
 59e5705:	48 89 c1             	mov    %rax,%rcx
 59e5708:	48 b8 f0 82 42 5e 00 	movabs $0x5e4282f0,%rax
 59e570f:	00 00 00 
 59e5712:	48 89 08             	mov    %rcx,(%rax)
 59e5715:	48 89 c8             	mov    %rcx,%rax
 59e5718:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e571f:	00 00 00 
 59e5722:	48 89 c1             	mov    %rax,%rcx
 59e5725:	48 b8 70 83 42 5e 00 	movabs $0x5e428370,%rax
 59e572c:	00 00 00 
 59e572f:	48 89 08             	mov    %rcx,(%rax)
 59e5732:	48 89 c8             	mov    %rcx,%rax
 59e5735:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e573c:	00 00 00 
 59e573f:	48 89 c1             	mov    %rax,%rcx
 59e5742:	48 b8 f0 83 42 5e 00 	movabs $0x5e4283f0,%rax
 59e5749:	00 00 00 
 59e574c:	48 89 08             	mov    %rcx,(%rax)
 59e574f:	48 89 c8             	mov    %rcx,%rax
 59e5752:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5759:	00 00 00 
 59e575c:	48 89 c1             	mov    %rax,%rcx
 59e575f:	48 b8 70 84 42 5e 00 	movabs $0x5e428470,%rax
 59e5766:	00 00 00 
 59e5769:	48 89 08             	mov    %rcx,(%rax)
 59e576c:	48 89 c8             	mov    %rcx,%rax
 59e576f:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5776:	00 00 00 
 59e5779:	48 89 c1             	mov    %rax,%rcx
 59e577c:	48 b8 f0 84 42 5e 00 	movabs $0x5e4284f0,%rax
 59e5783:	00 00 00 
 59e5786:	48 89 08             	mov    %rcx,(%rax)
 59e5789:	48 89 c8             	mov    %rcx,%rax
 59e578c:	59                   	pop    %rcx
 59e578d:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
 59e5794:	00 
 59e5795:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
 59e579c:	00 
 59e579d:	e8 4f 3e 2e fc       	call   0x1cc95f1
 59e57a2:	48 83 c4 10          	add    $0x10,%rsp
 59e57a6:	48 85 c0             	test   %rax,%rax
 59e57a9:	0f 84 10 00 00 00    	je     0x59e57bf
 59e57af:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 59e57b6:	48 81 c4 80 09 00 00 	add    $0x980,%rsp
 59e57bd:	5d                   	pop    %rbp
 59e57be:	c3                   	ret
 59e57bf:	90                   	nop
 59e57c0:	e9 43 00 00 00       	jmp    0x59e5808
 59e57c5:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 59e57cc:	50                   	push   %rax
 59e57cd:	31 c0                	xor    %eax,%eax
 59e57cf:	59                   	pop    %rcx
 59e57d0:	48 39 c1             	cmp    %rax,%rcx
 59e57d3:	0f 9d c0             	setge  %al
 59e57d6:	48 0f b6 c0          	movzbq %al,%rax
 59e57da:	48 85 c0             	test   %rax,%rax
 59e57dd:	0f 84 16 00 00 00    	je     0x59e57f9
 59e57e3:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e57ea:	00 00 00 
 59e57ed:	48 f7 d8             	neg    %rax
 59e57f0:	48 81 c4 80 09 00 00 	add    $0x980,%rsp
 59e57f7:	5d                   	pop    %rbp
 59e57f8:	c3                   	ret
 59e57f9:	90                   	nop
 59e57fa:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 59e5801:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
 59e5808:	90                   	nop
 59e5809:	48 8b 85 e0 ff ff ff 	mov    -0x20(%rbp),%rax
 59e5810:	50                   	push   %rax
 59e5811:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e5818:	00 00 00 
 59e581b:	59                   	pop    %rcx
 59e581c:	48 01 c8             	add    %rcx,%rax
 59e581f:	48 89 85 e0 ff ff ff 	mov    %rax,-0x20(%rbp)
 59e5826:	e9 91 f9 ff ff       	jmp    0x59e51bc
 59e582b:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
 59e5832:	48 8b 80 80 01 00 00 	mov    0x180(%rax),%rax
 59e5839:	50                   	push   %rax
 59e583a:	31 c0                	xor    %eax,%eax
 59e583c:	59                   	pop    %rcx
 59e583d:	48 39 c1             	cmp    %rax,%rcx
 59e5840:	0f 9f c0             	setg   %al
 59e5843:	48 0f b6 c0          	movzbq %al,%rax
 59e5847:	48 85 c0             	test   %rax,%rax
 59e584a:	0f 84 12 00 00 00    	je     0x59e5862
 59e5850:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e5857:	00 00 00 
 59e585a:	48 f7 d8             	neg    %rax
 59e585d:	e9 07 00 00 00       	jmp    0x59e5869
 59e5862:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
 59e5869:	48 81 c4 80 09 00 00 	add    $0x980,%rsp
 59e5870:	5d                   	pop    %rbp
 59e5871:	c3                   	ret
 59e5872:	55                   	push   %rbp
 59e5873:	48 89 e5             	mov    %rsp,%rbp
 59e5876:	48 81 ec 00 08 00 00 	sub    $0x800,%rsp
 59e587d:	48 89 e8             	mov    %rbp,%rax
 59e5880:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 59e5886:	48 39 e0             	cmp    %rsp,%rax
 59e5889:	76 05                	jbe    0x59e5890
 59e588b:	80 08 00             	orb    $0x0,(%rax)
 59e588e:	eb f0                	jmp    0x59e5880
 59e5890:	48 89 f8             	mov    %rdi,%rax
 59e5893:	48 89 85 f8 ff ff ff 	mov    %rax,-0x8(%rbp)
 59e589a:	48 89 f0             	mov    %rsi,%rax
 59e589d:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 59e58a4:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
 59e58ab:	50                   	push   %rax
 59e58ac:	31 c0                	xor    %eax,%eax
 59e58ae:	59                   	pop    %rcx
 59e58af:	48 39 c1             	cmp    %rax,%rcx
 59e58b2:	0f 9c c0             	setl   %al
 59e58b5:	48 0f b6 c0          	movzbq %al,%rax
 59e58b9:	48 85 c0             	test   %rax,%rax
 59e58bc:	0f 85 2d 00 00 00    	jne    0x59e58ef
 59e58c2:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
 59e58c9:	50                   	push   %rax
 59e58ca:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
 59e58d1:	48 8b 80 68 00 00 00 	mov    0x68(%rax),%rax
 59e58d8:	48 8b 80 00 80 00 00 	mov    0x8000(%rax),%rax
 59e58df:	59                   	pop    %rcx
 59e58e0:	48 39 c1             	cmp    %rax,%rcx
 59e58e3:	0f 9d c0             	setge  %al
 59e58e6:	48 0f b6 c0          	movzbq %al,%rax
 59e58ea:	e9 0a 00 00 00       	jmp    0x59e58f9
 59e58ef:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e58f6:	00 00 00 
 59e58f9:	48 85 c0             	test   %rax,%rax
 59e58fc:	0f 84 0b 00 00 00    	je     0x59e590d
 59e5902:	31 c0                	xor    %eax,%eax
 59e5904:	48 81 c4 00 08 00 00 	add    $0x800,%rsp
 59e590b:	5d                   	pop    %rbp
 59e590c:	c3                   	ret
 59e590d:	90                   	nop
 59e590e:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
 59e5915:	48 8b 80 68 00 00 00 	mov    0x68(%rax),%rax
 59e591c:	50                   	push   %rax
 59e591d:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
 59e5924:	48 89 c2             	mov    %rax,%rdx
 59e5927:	59                   	pop    %rcx
 59e5928:	48 8b 04 d1          	mov    (%rcx,%rdx,8),%rax
 59e592c:	48 89 c6             	mov    %rax,%rsi
 59e592f:	48 8d bd 60 fe ff ff 	lea    -0x1a0(%rbp),%rdi
 59e5936:	b9 31 00 00 00       	mov    $0x31,%ecx
 59e593b:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e593e:	48 8d 85 60 fe ff ff 	lea    -0x1a0(%rbp),%rax
 59e5945:	48 89 85 e8 ff ff ff 	mov    %rax,-0x18(%rbp)
 59e594c:	31 c0                	xor    %eax,%eax
 59e594e:	48 89 85 58 fe ff ff 	mov    %rax,-0x1a8(%rbp)
 59e5955:	48 8b 85 58 fe ff ff 	mov    -0x1a8(%rbp),%rax
 59e595c:	50                   	push   %rax
 59e595d:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
 59e5964:	48 8b 80 68 00 00 00 	mov    0x68(%rax),%rax
 59e596b:	48 8b 80 00 80 00 00 	mov    0x8000(%rax),%rax
 59e5972:	59                   	pop    %rcx
 59e5973:	48 39 c1             	cmp    %rax,%rcx
 59e5976:	0f 9c c0             	setl   %al
 59e5979:	48 0f b6 c0          	movzbq %al,%rax
 59e597d:	48 85 c0             	test   %rax,%rax
 59e5980:	0f 84 90 03 00 00    	je     0x59e5d16
 59e5986:	48 8b 85 f8 ff ff ff 	mov    -0x8(%rbp),%rax
 59e598d:	48 8b 80 68 00 00 00 	mov    0x68(%rax),%rax
 59e5994:	50                   	push   %rax
 59e5995:	48 8b 85 58 fe ff ff 	mov    -0x1a8(%rbp),%rax
 59e599c:	48 89 c2             	mov    %rax,%rdx
 59e599f:	59                   	pop    %rcx
 59e59a0:	48 8b 04 d1          	mov    (%rcx,%rdx,8),%rax
 59e59a4:	48 89 c6             	mov    %rax,%rsi
 59e59a7:	48 8d bd 28 fb ff ff 	lea    -0x4d8(%rbp),%rdi
 59e59ae:	b9 65 00 00 00       	mov    $0x65,%ecx
 59e59b3:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e59b6:	48 8d 85 28 fb ff ff 	lea    -0x4d8(%rbp),%rax
 59e59bd:	48 89 85 50 fe ff ff 	mov    %rax,-0x1b0(%rbp)
 59e59c4:	48 8b 85 50 fe ff ff 	mov    -0x1b0(%rbp),%rax
 59e59cb:	48 89 c6             	mov    %rax,%rsi
 59e59ce:	48 8d bd a0 f9 ff ff 	lea    -0x660(%rbp),%rdi
 59e59d5:	b9 31 00 00 00       	mov    $0x31,%ecx
 59e59da:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e59dd:	48 8d 85 a0 f9 ff ff 	lea    -0x660(%rbp),%rax
 59e59e4:	50                   	push   %rax
 59e59e5:	48 8b 85 e8 ff ff ff 	mov    -0x18(%rbp),%rax
 59e59ec:	48 89 c6             	mov    %rax,%rsi
 59e59ef:	48 8d bd 18 f8 ff ff 	lea    -0x7e8(%rbp),%rdi
 59e59f6:	b9 31 00 00 00       	mov    $0x31,%ecx
 59e59fb:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
 59e59fe:	48 8d 85 18 f8 ff ff 	lea    -0x7e8(%rbp),%rax
 59e5a05:	50                   	push   %rax
 59e5a06:	66 90                	xchg   %ax,%ax
 59e5a08:	51                   	push   %rcx
 59e5a09:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5a10:	00 00 00 
 59e5a13:	48 89 c1             	mov    %rax,%rcx
 59e5a16:	48 b8 20 80 42 5e 00 	movabs $0x5e428020,%rax
 59e5a1d:	00 00 00 
 59e5a20:	48 89 08             	mov    %rcx,(%rax)
 59e5a23:	48 89 c8             	mov    %rcx,%rax
 59e5a26:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5a2d:	00 00 00 
 59e5a30:	48 89 c1             	mov    %rax,%rcx
 59e5a33:	48 b8 28 80 42 5e 00 	movabs $0x5e428028,%rax
 59e5a3a:	00 00 00 
 59e5a3d:	48 89 08             	mov    %rcx,(%rax)
 59e5a40:	48 89 c8             	mov    %rcx,%rax
 59e5a43:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5a4a:	00 00 00 
 59e5a4d:	48 89 c1             	mov    %rax,%rcx
 59e5a50:	48 b8 e8 80 42 5e 00 	movabs $0x5e4280e8,%rax
 59e5a57:	00 00 00 
 59e5a5a:	48 89 08             	mov    %rcx,(%rax)
 59e5a5d:	48 89 c8             	mov    %rcx,%rax
 59e5a60:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5a67:	00 00 00 
 59e5a6a:	48 89 c1             	mov    %rax,%rcx
 59e5a6d:	48 b8 68 81 42 5e 00 	movabs $0x5e428168,%rax
 59e5a74:	00 00 00 
 59e5a77:	48 89 08             	mov    %rcx,(%rax)
 59e5a7a:	48 89 c8             	mov    %rcx,%rax
 59e5a7d:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5a84:	00 00 00 
 59e5a87:	48 89 c1             	mov    %rax,%rcx
 59e5a8a:	48 b8 e8 81 42 5e 00 	movabs $0x5e4281e8,%rax
 59e5a91:	00 00 00 
 59e5a94:	48 89 08             	mov    %rcx,(%rax)
 59e5a97:	48 89 c8             	mov    %rcx,%rax
 59e5a9a:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5aa1:	00 00 00 
 59e5aa4:	48 89 c1             	mov    %rax,%rcx
 59e5aa7:	48 b8 68 82 42 5e 00 	movabs $0x5e428268,%rax
 59e5aae:	00 00 00 
 59e5ab1:	48 89 08             	mov    %rcx,(%rax)
 59e5ab4:	48 89 c8             	mov    %rcx,%rax
 59e5ab7:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5abe:	00 00 00 
 59e5ac1:	48 89 c1             	mov    %rax,%rcx
 59e5ac4:	48 b8 e8 82 42 5e 00 	movabs $0x5e4282e8,%rax
 59e5acb:	00 00 00 
 59e5ace:	48 89 08             	mov    %rcx,(%rax)
 59e5ad1:	48 89 c8             	mov    %rcx,%rax
 59e5ad4:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5adb:	00 00 00 
 59e5ade:	48 89 c1             	mov    %rax,%rcx
 59e5ae1:	48 b8 68 83 42 5e 00 	movabs $0x5e428368,%rax
 59e5ae8:	00 00 00 
 59e5aeb:	48 89 08             	mov    %rcx,(%rax)
 59e5aee:	48 89 c8             	mov    %rcx,%rax
 59e5af1:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5af8:	00 00 00 
 59e5afb:	48 89 c1             	mov    %rax,%rcx
 59e5afe:	48 b8 e8 83 42 5e 00 	movabs $0x5e4283e8,%rax
 59e5b05:	00 00 00 
 59e5b08:	48 89 08             	mov    %rcx,(%rax)
 59e5b0b:	48 89 c8             	mov    %rcx,%rax
 59e5b0e:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5b15:	00 00 00 
 59e5b18:	48 89 c1             	mov    %rax,%rcx
 59e5b1b:	48 b8 68 84 42 5e 00 	movabs $0x5e428468,%rax
 59e5b22:	00 00 00 
 59e5b25:	48 89 08             	mov    %rcx,(%rax)
 59e5b28:	48 89 c8             	mov    %rcx,%rax
 59e5b2b:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5b32:	00 00 00 
 59e5b35:	48 89 c1             	mov    %rax,%rcx
 59e5b38:	48 b8 e8 84 42 5e 00 	movabs $0x5e4284e8,%rax
 59e5b3f:	00 00 00 
 59e5b42:	48 89 08             	mov    %rcx,(%rax)
 59e5b45:	48 89 c8             	mov    %rcx,%rax
 59e5b48:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5b4f:	00 00 00 
 59e5b52:	48 89 c1             	mov    %rax,%rcx
 59e5b55:	48 b8 f0 80 42 5e 00 	movabs $0x5e4280f0,%rax
 59e5b5c:	00 00 00 
 59e5b5f:	48 89 08             	mov    %rcx,(%rax)
 59e5b62:	48 89 c8             	mov    %rcx,%rax
 59e5b65:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5b6c:	00 00 00 
 59e5b6f:	48 89 c1             	mov    %rax,%rcx
 59e5b72:	48 b8 70 81 42 5e 00 	movabs $0x5e428170,%rax
 59e5b79:	00 00 00 
 59e5b7c:	48 89 08             	mov    %rcx,(%rax)
 59e5b7f:	48 89 c8             	mov    %rcx,%rax
 59e5b82:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5b89:	00 00 00 
 59e5b8c:	48 89 c1             	mov    %rax,%rcx
 59e5b8f:	48 b8 f0 81 42 5e 00 	movabs $0x5e4281f0,%rax
 59e5b96:	00 00 00 
 59e5b99:	48 89 08             	mov    %rcx,(%rax)
 59e5b9c:	48 89 c8             	mov    %rcx,%rax
 59e5b9f:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5ba6:	00 00 00 
 59e5ba9:	48 89 c1             	mov    %rax,%rcx
 59e5bac:	48 b8 70 82 42 5e 00 	movabs $0x5e428270,%rax
 59e5bb3:	00 00 00 
 59e5bb6:	48 89 08             	mov    %rcx,(%rax)
 59e5bb9:	48 89 c8             	mov    %rcx,%rax
 59e5bbc:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5bc3:	00 00 00 
 59e5bc6:	48 89 c1             	mov    %rax,%rcx
 59e5bc9:	48 b8 f0 82 42 5e 00 	movabs $0x5e4282f0,%rax
 59e5bd0:	00 00 00 
 59e5bd3:	48 89 08             	mov    %rcx,(%rax)
 59e5bd6:	48 89 c8             	mov    %rcx,%rax
 59e5bd9:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5be0:	00 00 00 
 59e5be3:	48 89 c1             	mov    %rax,%rcx
 59e5be6:	48 b8 70 83 42 5e 00 	movabs $0x5e428370,%rax
 59e5bed:	00 00 00 
 59e5bf0:	48 89 08             	mov    %rcx,(%rax)
 59e5bf3:	48 89 c8             	mov    %rcx,%rax
 59e5bf6:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5bfd:	00 00 00 
 59e5c00:	48 89 c1             	mov    %rax,%rcx
 59e5c03:	48 b8 f0 83 42 5e 00 	movabs $0x5e4283f0,%rax
 59e5c0a:	00 00 00 
 59e5c0d:	48 89 08             	mov    %rcx,(%rax)
 59e5c10:	48 89 c8             	mov    %rcx,%rax
 59e5c13:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5c1a:	00 00 00 
 59e5c1d:	48 89 c1             	mov    %rax,%rcx
 59e5c20:	48 b8 70 84 42 5e 00 	movabs $0x5e428470,%rax
 59e5c27:	00 00 00 
 59e5c2a:	48 89 08             	mov    %rcx,(%rax)
 59e5c2d:	48 89 c8             	mov    %rcx,%rax
 59e5c30:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5c37:	00 00 00 
 59e5c3a:	48 89 c1             	mov    %rax,%rcx
 59e5c3d:	48 b8 f0 84 42 5e 00 	movabs $0x5e4284f0,%rax
 59e5c44:	00 00 00 
 59e5c47:	48 89 08             	mov    %rcx,(%rax)
 59e5c4a:	48 89 c8             	mov    %rcx,%rax
 59e5c4d:	59                   	pop    %rcx
 59e5c4e:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
 59e5c55:	00 
 59e5c56:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
 59e5c5d:	00 
 59e5c5e:	e8 8e 39 2e fc       	call   0x1cc95f1
 59e5c63:	48 83 c4 10          	add    $0x10,%rsp
 59e5c67:	48 85 c0             	test   %rax,%rax
 59e5c6a:	0f 84 83 00 00 00    	je     0x59e5cf3
 59e5c70:	48 8b 85 50 fe ff ff 	mov    -0x1b0(%rbp),%rax
 59e5c77:	48 8d 80 18 03 00 00 	lea    0x318(%rax),%rax
 59e5c7e:	48 89 85 10 f8 ff ff 	mov    %rax,-0x7f0(%rbp)
 59e5c85:	48 8b 85 10 f8 ff ff 	mov    -0x7f0(%rbp),%rax
 59e5c8c:	48 8b 00             	mov    (%rax),%rax
 59e5c8f:	50                   	push   %rax
 59e5c90:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e5c97:	00 00 00 
 59e5c9a:	59                   	pop    %rcx
 59e5c9b:	48 39 c1             	cmp    %rax,%rcx
 59e5c9e:	0f 85 2a 00 00 00    	jne    0x59e5cce
 59e5ca4:	48 8b 85 10 f8 ff ff 	mov    -0x7f0(%rbp),%rax
 59e5cab:	48 8b 40 08          	mov    0x8(%rax),%rax
 59e5caf:	48 89 85 08 f8 ff ff 	mov    %rax,-0x7f8(%rbp)
 59e5cb6:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e5cbd:	00 00 00 
 59e5cc0:	48 81 c4 00 08 00 00 	add    $0x800,%rsp
 59e5cc7:	5d                   	pop    %rbp
 59e5cc8:	c3                   	ret
 59e5cc9:	e9 24 00 00 00       	jmp    0x59e5cf2
 59e5cce:	48 8b 85 10 f8 ff ff 	mov    -0x7f0(%rbp),%rax
 59e5cd5:	48 8b 00             	mov    (%rax),%rax
 59e5cd8:	50                   	push   %rax
 59e5cd9:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
 59e5ce0:	00 00 00 
 59e5ce3:	59                   	pop    %rcx
 59e5ce4:	48 39 c1             	cmp    %rax,%rcx
 59e5ce7:	0f 85 05 00 00 00    	jne    0x59e5cf2
 59e5ced:	e9 00 00 00 00       	jmp    0x59e5cf2
 59e5cf2:	90                   	nop
 59e5cf3:	90                   	nop
 59e5cf4:	48 8b 85 58 fe ff ff 	mov    -0x1a8(%rbp),%rax
 59e5cfb:	50                   	push   %rax
 59e5cfc:	48 b8 01 00 00 00 00 	movabs $0x1,%rax
 59e5d03:	00 00 00 
 59e5d06:	59                   	pop    %rcx
 59e5d07:	48 01 c8             	add    %rcx,%rax
 59e5d0a:	48 89 85 58 fe ff ff 	mov    %rax,-0x1a8(%rbp)
 59e5d11:	e9 3f fc ff ff       	jmp    0x59e5955
 59e5d16:	31 c0                	xor    %eax,%eax
 59e5d18:	48 81 c4 00 08 00 00 	add    $0x800,%rsp
 59e5d1f:	5d                   	pop    %rbp
 59e5d20:	c3                   	ret
 59e5d21:	55                   	push   %rbp
 59e5d22:	48 89 e5             	mov    %rsp,%rbp
 59e5d25:	48 81 ec a0 12 00 00 	sub    $0x12a0,%rsp
 59e5d2c:	48 89 e8             	mov    %rbp,%rax
 59e5d2f:	48 2d 00 10 00 00    	sub    $0x1000,%rax
 59e5d35:	48 39 e0             	cmp    %rsp,%rax
 59e5d38:	76 05                	jbe    0x59e5d3f
 59e5d3a:	80 08 00             	orb    $0x0,(%rax)
 59e5d3d:	eb f0                	jmp    0x59e5d2f
 59e5d3f:	48 89 f8             	mov    %rdi,%rax
 59e5d42:	48 89 85 f8 ff ff ff 	mov    %rax,-0x8(%rbp)
 59e5d49:	48 89 f0             	mov    %rsi,%rax
 59e5d4c:	48 89 85 f0 ff ff ff 	mov    %rax,-0x10(%rbp)
 59e5d53:	48 8b 85 f0 ff ff ff 	mov    -0x10(%rbp),%rax
 59e5d5a:	48 8d 80 80 03 00 00 	lea    0x380(%rax),%rax
 59e5d61:	48                   	rex.W
 59e5d62:	89                   	.byte 0x89

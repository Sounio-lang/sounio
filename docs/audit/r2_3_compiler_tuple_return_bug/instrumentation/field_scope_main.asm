    14e7:	55                   	push   %rbp
    14e8:	48 89 e5             	mov    %rsp,%rbp
    14eb:	48 81 ec a0 01 00 00 	sub    $0x1a0,%rsp
    14f2:	48 b8 a4 26 35 01 00 	movabs $0x13526a4,%rax
    14f9:	00 00 00 
    14fc:	50                   	push   %rax
    14fd:	51                   	push   %rcx
    14fe:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    1505:	00 00 00 
    1508:	48 89 c1             	mov    %rax,%rcx
    150b:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    1512:	00 00 00 
    1515:	48 89 08             	mov    %rcx,(%rax)
    1518:	48 89 c8             	mov    %rcx,%rax
    151b:	59                   	pop    %rcx
    151c:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
    1523:	00 
    1524:	48 8d 85 e0 ff ff ff 	lea    -0x20(%rbp),%rax
    152b:	48 89 c7             	mov    %rax,%rdi
    152e:	e8 cd fa ff ff       	call   0x1000
    1533:	48 83 c4 08          	add    $0x8,%rsp
    1537:	48 89 c2             	mov    %rax,%rdx
    153a:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    1541:	48 89 85 b8 ff ff ff 	mov    %rax,-0x48(%rbp)
    1548:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    154f:	48 89 85 c0 ff ff ff 	mov    %rax,-0x40(%rbp)
    1556:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    155d:	48 89 85 c8 ff ff ff 	mov    %rax,-0x38(%rbp)
    1564:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    156b:	48 89 85 d0 ff ff ff 	mov    %rax,-0x30(%rbp)
    1572:	48 8d 85 b8 ff ff ff 	lea    -0x48(%rbp),%rax
    1579:	48 89 85 d8 ff ff ff 	mov    %rax,-0x28(%rbp)
    1580:	48 8b 85 d8 ff ff ff 	mov    -0x28(%rbp),%rax
    1587:	48 89 c2             	mov    %rax,%rdx
    158a:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    1591:	48 89 85 98 ff ff ff 	mov    %rax,-0x68(%rbp)
    1598:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    159f:	48 89 85 a0 ff ff ff 	mov    %rax,-0x60(%rbp)
    15a6:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    15ad:	48 89 85 a8 ff ff ff 	mov    %rax,-0x58(%rbp)
    15b4:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    15bb:	48 89 85 b0 ff ff ff 	mov    %rax,-0x50(%rbp)
    15c2:	48 8d 85 98 ff ff ff 	lea    -0x68(%rbp),%rax
    15c9:	50                   	push   %rax
    15ca:	51                   	push   %rcx
    15cb:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    15d2:	00 00 00 
    15d5:	48 89 c1             	mov    %rax,%rcx
    15d8:	48 b8 a0 00 00 10 00 	movabs $0x100000a0,%rax
    15df:	00 00 00 
    15e2:	48 89 08             	mov    %rcx,(%rax)
    15e5:	48 89 c8             	mov    %rcx,%rax
    15e8:	59                   	pop    %rcx
    15e9:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
    15f0:	00 
    15f1:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
    15f8:	48 89 c7             	mov    %rax,%rdi
    15fb:	e8 ea fc ff ff       	call   0x12ea
    1600:	48 83 c4 08          	add    $0x8,%rsp
    1604:	48 89 85 68 ff ff ff 	mov    %rax,-0x98(%rbp)
    160b:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    1612:	48 89 c2             	mov    %rax,%rdx
    1615:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    161c:	48 89 85 48 ff ff ff 	mov    %rax,-0xb8(%rbp)
    1623:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    162a:	48 89 85 50 ff ff ff 	mov    %rax,-0xb0(%rbp)
    1631:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    1638:	48 89 85 58 ff ff ff 	mov    %rax,-0xa8(%rbp)
    163f:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    1646:	48 89 85 60 ff ff ff 	mov    %rax,-0xa0(%rbp)
    164d:	48 8d 85 48 ff ff ff 	lea    -0xb8(%rbp),%rax
    1654:	48 8b 00             	mov    (%rax),%rax
    1657:	48 83 ec 20          	sub    $0x20,%rsp
    165b:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    1660:	c6 01 0a             	movb   $0xa,(%rcx)
    1663:	48 ff c9             	dec    %rcx
    1666:	4d 31 c0             	xor    %r8,%r8
    1669:	48 85 c0             	test   %rax,%rax
    166c:	79 06                	jns    0x1674
    166e:	48 f7 d8             	neg    %rax
    1671:	49 ff c0             	inc    %r8
    1674:	48 31 d2             	xor    %rdx,%rdx
    1677:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    167e:	48 f7 f6             	div    %rsi
    1681:	80 c2 30             	add    $0x30,%dl
    1684:	88 11                	mov    %dl,(%rcx)
    1686:	48 ff c9             	dec    %rcx
    1689:	48 85 c0             	test   %rax,%rax
    168c:	0f 85 e2 ff ff ff    	jne    0x1674
    1692:	4d 85 c0             	test   %r8,%r8
    1695:	74 06                	je     0x169d
    1697:	c6 01 2d             	movb   $0x2d,(%rcx)
    169a:	48 ff c9             	dec    %rcx
    169d:	48 ff c1             	inc    %rcx
    16a0:	48 89 ce             	mov    %rcx,%rsi
    16a3:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    16a8:	48 29 ca             	sub    %rcx,%rdx
    16ab:	b8 01 00 00 00       	mov    $0x1,%eax
    16b0:	bf 01 00 00 00       	mov    $0x1,%edi
    16b5:	0f 05                	syscall
    16b7:	48 83 c4 20          	add    $0x20,%rsp
    16bb:	31 c0                	xor    %eax,%eax
    16bd:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    16c4:	48 89 c2             	mov    %rax,%rdx
    16c7:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    16ce:	48 89 85 28 ff ff ff 	mov    %rax,-0xd8(%rbp)
    16d5:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    16dc:	48 89 85 30 ff ff ff 	mov    %rax,-0xd0(%rbp)
    16e3:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    16ea:	48 89 85 38 ff ff ff 	mov    %rax,-0xc8(%rbp)
    16f1:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    16f8:	48 89 85 40 ff ff ff 	mov    %rax,-0xc0(%rbp)
    16ff:	48 8d 85 28 ff ff ff 	lea    -0xd8(%rbp),%rax
    1706:	48 8b 80 08 00 00 00 	mov    0x8(%rax),%rax
    170d:	48 83 ec 20          	sub    $0x20,%rsp
    1711:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    1716:	c6 01 0a             	movb   $0xa,(%rcx)
    1719:	48 ff c9             	dec    %rcx
    171c:	4d 31 c0             	xor    %r8,%r8
    171f:	48 85 c0             	test   %rax,%rax
    1722:	79 06                	jns    0x172a
    1724:	48 f7 d8             	neg    %rax
    1727:	49 ff c0             	inc    %r8
    172a:	48 31 d2             	xor    %rdx,%rdx
    172d:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    1734:	48 f7 f6             	div    %rsi
    1737:	80 c2 30             	add    $0x30,%dl
    173a:	88 11                	mov    %dl,(%rcx)
    173c:	48 ff c9             	dec    %rcx
    173f:	48 85 c0             	test   %rax,%rax
    1742:	0f 85 e2 ff ff ff    	jne    0x172a
    1748:	4d 85 c0             	test   %r8,%r8
    174b:	74 06                	je     0x1753
    174d:	c6 01 2d             	movb   $0x2d,(%rcx)
    1750:	48 ff c9             	dec    %rcx
    1753:	48 ff c1             	inc    %rcx
    1756:	48 89 ce             	mov    %rcx,%rsi
    1759:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    175e:	48 29 ca             	sub    %rcx,%rdx
    1761:	b8 01 00 00 00       	mov    $0x1,%eax
    1766:	bf 01 00 00 00       	mov    $0x1,%edi
    176b:	0f 05                	syscall
    176d:	48 83 c4 20          	add    $0x20,%rsp
    1771:	31 c0                	xor    %eax,%eax
    1773:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    177a:	48 89 c2             	mov    %rax,%rdx
    177d:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    1784:	48 89 85 08 ff ff ff 	mov    %rax,-0xf8(%rbp)
    178b:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    1792:	48 89 85 10 ff ff ff 	mov    %rax,-0xf0(%rbp)
    1799:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    17a0:	48 89 85 18 ff ff ff 	mov    %rax,-0xe8(%rbp)
    17a7:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    17ae:	48 89 85 20 ff ff ff 	mov    %rax,-0xe0(%rbp)
    17b5:	48 8d 85 08 ff ff ff 	lea    -0xf8(%rbp),%rax
    17bc:	48 8b 80 10 00 00 00 	mov    0x10(%rax),%rax
    17c3:	48 83 ec 20          	sub    $0x20,%rsp
    17c7:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    17cc:	c6 01 0a             	movb   $0xa,(%rcx)
    17cf:	48 ff c9             	dec    %rcx
    17d2:	4d 31 c0             	xor    %r8,%r8
    17d5:	48 85 c0             	test   %rax,%rax
    17d8:	79 06                	jns    0x17e0
    17da:	48 f7 d8             	neg    %rax
    17dd:	49 ff c0             	inc    %r8
    17e0:	48 31 d2             	xor    %rdx,%rdx
    17e3:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    17ea:	48 f7 f6             	div    %rsi
    17ed:	80 c2 30             	add    $0x30,%dl
    17f0:	88 11                	mov    %dl,(%rcx)
    17f2:	48 ff c9             	dec    %rcx
    17f5:	48 85 c0             	test   %rax,%rax
    17f8:	0f 85 e2 ff ff ff    	jne    0x17e0
    17fe:	4d 85 c0             	test   %r8,%r8
    1801:	74 06                	je     0x1809
    1803:	c6 01 2d             	movb   $0x2d,(%rcx)
    1806:	48 ff c9             	dec    %rcx
    1809:	48 ff c1             	inc    %rcx
    180c:	48 89 ce             	mov    %rcx,%rsi
    180f:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    1814:	48 29 ca             	sub    %rcx,%rdx
    1817:	b8 01 00 00 00       	mov    $0x1,%eax
    181c:	bf 01 00 00 00       	mov    $0x1,%edi
    1821:	0f 05                	syscall
    1823:	48 83 c4 20          	add    $0x20,%rsp
    1827:	31 c0                	xor    %eax,%eax
    1829:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    1830:	48 89 c2             	mov    %rax,%rdx
    1833:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    183a:	48 89 85 e8 fe ff ff 	mov    %rax,-0x118(%rbp)
    1841:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    1848:	48 89 85 f0 fe ff ff 	mov    %rax,-0x110(%rbp)
    184f:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    1856:	48 89 85 f8 fe ff ff 	mov    %rax,-0x108(%rbp)
    185d:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    1864:	48 89 85 00 ff ff ff 	mov    %rax,-0x100(%rbp)
    186b:	48 8d 85 e8 fe ff ff 	lea    -0x118(%rbp),%rax
    1872:	48 8b 80 18 00 00 00 	mov    0x18(%rax),%rax
    1879:	48 83 ec 20          	sub    $0x20,%rsp
    187d:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    1882:	c6 01 0a             	movb   $0xa,(%rcx)
    1885:	48 ff c9             	dec    %rcx
    1888:	4d 31 c0             	xor    %r8,%r8
    188b:	48 85 c0             	test   %rax,%rax
    188e:	79 06                	jns    0x1896
    1890:	48 f7 d8             	neg    %rax
    1893:	49 ff c0             	inc    %r8
    1896:	48 31 d2             	xor    %rdx,%rdx
    1899:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    18a0:	48 f7 f6             	div    %rsi
    18a3:	80 c2 30             	add    $0x30,%dl
    18a6:	88 11                	mov    %dl,(%rcx)
    18a8:	48 ff c9             	dec    %rcx
    18ab:	48 85 c0             	test   %rax,%rax
    18ae:	0f 85 e2 ff ff ff    	jne    0x1896
    18b4:	4d 85 c0             	test   %r8,%r8
    18b7:	74 06                	je     0x18bf
    18b9:	c6 01 2d             	movb   $0x2d,(%rcx)
    18bc:	48 ff c9             	dec    %rcx
    18bf:	48 ff c1             	inc    %rcx
    18c2:	48 89 ce             	mov    %rcx,%rsi
    18c5:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    18ca:	48 29 ca             	sub    %rcx,%rdx
    18cd:	b8 01 00 00 00       	mov    $0x1,%eax
    18d2:	bf 01 00 00 00       	mov    $0x1,%edi
    18d7:	0f 05                	syscall
    18d9:	48 83 c4 20          	add    $0x20,%rsp
    18dd:	31 c0                	xor    %eax,%eax
    18df:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    18e6:	48 05 20 00 00 00    	add    $0x20,%rax
    18ec:	48 8b 00             	mov    (%rax),%rax
    18ef:	50                   	push   %rax
    18f0:	48 b8 06 00 00 00 00 	movabs $0x6,%rax
    18f7:	00 00 00 
    18fa:	50                   	push   %rax
    18fb:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
    1902:	00 
    1903:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
    190a:	00 
    190b:	e8 5d 5f 00 00       	call   0x786d
    1910:	48 83 c4 10          	add    $0x10,%rsp
    1914:	31 c0                	xor    %eax,%eax
    1916:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    191d:	48 89 c2             	mov    %rax,%rdx
    1920:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    1927:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    192e:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    1935:	48 89 85 d0 fe ff ff 	mov    %rax,-0x130(%rbp)
    193c:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    1943:	48 89 85 d8 fe ff ff 	mov    %rax,-0x128(%rbp)
    194a:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    1951:	48 89 85 e0 fe ff ff 	mov    %rax,-0x120(%rbp)
    1958:	48 8d 85 c8 fe ff ff 	lea    -0x138(%rbp),%rax
    195f:	48 8b 00             	mov    (%rax),%rax
    1962:	48 83 ec 20          	sub    $0x20,%rsp
    1966:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    196b:	c6 01 0a             	movb   $0xa,(%rcx)
    196e:	48 ff c9             	dec    %rcx
    1971:	4d 31 c0             	xor    %r8,%r8
    1974:	48 85 c0             	test   %rax,%rax
    1977:	79 06                	jns    0x197f
    1979:	48 f7 d8             	neg    %rax
    197c:	49 ff c0             	inc    %r8
    197f:	48 31 d2             	xor    %rdx,%rdx
    1982:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    1989:	48 f7 f6             	div    %rsi
    198c:	80 c2 30             	add    $0x30,%dl
    198f:	88 11                	mov    %dl,(%rcx)
    1991:	48 ff c9             	dec    %rcx
    1994:	48 85 c0             	test   %rax,%rax
    1997:	0f 85 e2 ff ff ff    	jne    0x197f
    199d:	4d 85 c0             	test   %r8,%r8
    19a0:	74 06                	je     0x19a8
    19a2:	c6 01 2d             	movb   $0x2d,(%rcx)
    19a5:	48 ff c9             	dec    %rcx
    19a8:	48 ff c1             	inc    %rcx
    19ab:	48 89 ce             	mov    %rcx,%rsi
    19ae:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    19b3:	48 29 ca             	sub    %rcx,%rdx
    19b6:	b8 01 00 00 00       	mov    $0x1,%eax
    19bb:	bf 01 00 00 00       	mov    $0x1,%edi
    19c0:	0f 05                	syscall
    19c2:	48 83 c4 20          	add    $0x20,%rsp
    19c6:	31 c0                	xor    %eax,%eax
    19c8:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    19cf:	48 89 c2             	mov    %rax,%rdx
    19d2:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    19d9:	48 89 85 a8 fe ff ff 	mov    %rax,-0x158(%rbp)
    19e0:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    19e7:	48 89 85 b0 fe ff ff 	mov    %rax,-0x150(%rbp)
    19ee:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    19f5:	48 89 85 b8 fe ff ff 	mov    %rax,-0x148(%rbp)
    19fc:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    1a03:	48 89 85 c0 fe ff ff 	mov    %rax,-0x140(%rbp)
    1a0a:	48 8d 85 a8 fe ff ff 	lea    -0x158(%rbp),%rax
    1a11:	48 8b 80 08 00 00 00 	mov    0x8(%rax),%rax
    1a18:	48 83 ec 20          	sub    $0x20,%rsp
    1a1c:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    1a21:	c6 01 0a             	movb   $0xa,(%rcx)
    1a24:	48 ff c9             	dec    %rcx
    1a27:	4d 31 c0             	xor    %r8,%r8
    1a2a:	48 85 c0             	test   %rax,%rax
    1a2d:	79 06                	jns    0x1a35
    1a2f:	48 f7 d8             	neg    %rax
    1a32:	49 ff c0             	inc    %r8
    1a35:	48 31 d2             	xor    %rdx,%rdx
    1a38:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    1a3f:	48 f7 f6             	div    %rsi
    1a42:	80 c2 30             	add    $0x30,%dl
    1a45:	88 11                	mov    %dl,(%rcx)
    1a47:	48 ff c9             	dec    %rcx
    1a4a:	48 85 c0             	test   %rax,%rax
    1a4d:	0f 85 e2 ff ff ff    	jne    0x1a35
    1a53:	4d 85 c0             	test   %r8,%r8
    1a56:	74 06                	je     0x1a5e
    1a58:	c6 01 2d             	movb   $0x2d,(%rcx)
    1a5b:	48 ff c9             	dec    %rcx
    1a5e:	48 ff c1             	inc    %rcx
    1a61:	48 89 ce             	mov    %rcx,%rsi
    1a64:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    1a69:	48 29 ca             	sub    %rcx,%rdx
    1a6c:	b8 01 00 00 00       	mov    $0x1,%eax
    1a71:	bf 01 00 00 00       	mov    $0x1,%edi
    1a76:	0f 05                	syscall
    1a78:	48 83 c4 20          	add    $0x20,%rsp
    1a7c:	31 c0                	xor    %eax,%eax
    1a7e:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    1a85:	48 89 c2             	mov    %rax,%rdx
    1a88:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    1a8f:	48 89 85 88 fe ff ff 	mov    %rax,-0x178(%rbp)
    1a96:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    1a9d:	48 89 85 90 fe ff ff 	mov    %rax,-0x170(%rbp)
    1aa4:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    1aab:	48 89 85 98 fe ff ff 	mov    %rax,-0x168(%rbp)
    1ab2:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    1ab9:	48 89 85 a0 fe ff ff 	mov    %rax,-0x160(%rbp)
    1ac0:	48 8d 85 88 fe ff ff 	lea    -0x178(%rbp),%rax
    1ac7:	48 8b 80 10 00 00 00 	mov    0x10(%rax),%rax
    1ace:	48 83 ec 20          	sub    $0x20,%rsp
    1ad2:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    1ad7:	c6 01 0a             	movb   $0xa,(%rcx)
    1ada:	48 ff c9             	dec    %rcx
    1add:	4d 31 c0             	xor    %r8,%r8
    1ae0:	48 85 c0             	test   %rax,%rax
    1ae3:	79 06                	jns    0x1aeb
    1ae5:	48 f7 d8             	neg    %rax
    1ae8:	49 ff c0             	inc    %r8
    1aeb:	48 31 d2             	xor    %rdx,%rdx
    1aee:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    1af5:	48 f7 f6             	div    %rsi
    1af8:	80 c2 30             	add    $0x30,%dl
    1afb:	88 11                	mov    %dl,(%rcx)
    1afd:	48 ff c9             	dec    %rcx
    1b00:	48 85 c0             	test   %rax,%rax
    1b03:	0f 85 e2 ff ff ff    	jne    0x1aeb
    1b09:	4d 85 c0             	test   %r8,%r8
    1b0c:	74 06                	je     0x1b14
    1b0e:	c6 01 2d             	movb   $0x2d,(%rcx)
    1b11:	48 ff c9             	dec    %rcx
    1b14:	48 ff c1             	inc    %rcx
    1b17:	48 89 ce             	mov    %rcx,%rsi
    1b1a:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    1b1f:	48 29 ca             	sub    %rcx,%rdx
    1b22:	b8 01 00 00 00       	mov    $0x1,%eax
    1b27:	bf 01 00 00 00       	mov    $0x1,%edi
    1b2c:	0f 05                	syscall
    1b2e:	48 83 c4 20          	add    $0x20,%rsp
    1b32:	31 c0                	xor    %eax,%eax
    1b34:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    1b3b:	48 89 c2             	mov    %rax,%rdx
    1b3e:	48 8b 82 00 00 00 00 	mov    0x0(%rdx),%rax
    1b45:	48 89 85 68 fe ff ff 	mov    %rax,-0x198(%rbp)
    1b4c:	48 8b 82 08 00 00 00 	mov    0x8(%rdx),%rax
    1b53:	48 89 85 70 fe ff ff 	mov    %rax,-0x190(%rbp)
    1b5a:	48 8b 82 10 00 00 00 	mov    0x10(%rdx),%rax
    1b61:	48 89 85 78 fe ff ff 	mov    %rax,-0x188(%rbp)
    1b68:	48 8b 82 18 00 00 00 	mov    0x18(%rdx),%rax
    1b6f:	48 89 85 80 fe ff ff 	mov    %rax,-0x180(%rbp)
    1b76:	48 8d 85 68 fe ff ff 	lea    -0x198(%rbp),%rax
    1b7d:	48 8b 80 18 00 00 00 	mov    0x18(%rax),%rax
    1b84:	48 83 ec 20          	sub    $0x20,%rsp
    1b88:	48 8d 4c 24 1f       	lea    0x1f(%rsp),%rcx
    1b8d:	c6 01 0a             	movb   $0xa,(%rcx)
    1b90:	48 ff c9             	dec    %rcx
    1b93:	4d 31 c0             	xor    %r8,%r8
    1b96:	48 85 c0             	test   %rax,%rax
    1b99:	79 06                	jns    0x1ba1
    1b9b:	48 f7 d8             	neg    %rax
    1b9e:	49 ff c0             	inc    %r8
    1ba1:	48 31 d2             	xor    %rdx,%rdx
    1ba4:	48 c7 c6 0a 00 00 00 	mov    $0xa,%rsi
    1bab:	48 f7 f6             	div    %rsi
    1bae:	80 c2 30             	add    $0x30,%dl
    1bb1:	88 11                	mov    %dl,(%rcx)
    1bb3:	48 ff c9             	dec    %rcx
    1bb6:	48 85 c0             	test   %rax,%rax
    1bb9:	0f 85 e2 ff ff ff    	jne    0x1ba1
    1bbf:	4d 85 c0             	test   %r8,%r8
    1bc2:	74 06                	je     0x1bca
    1bc4:	c6 01 2d             	movb   $0x2d,(%rcx)
    1bc7:	48 ff c9             	dec    %rcx
    1bca:	48 ff c1             	inc    %rcx
    1bcd:	48 89 ce             	mov    %rcx,%rsi
    1bd0:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    1bd5:	48 29 ca             	sub    %rcx,%rdx
    1bd8:	b8 01 00 00 00       	mov    $0x1,%eax
    1bdd:	bf 01 00 00 00       	mov    $0x1,%edi
    1be2:	0f 05                	syscall
    1be4:	48 83 c4 20          	add    $0x20,%rsp
    1be8:	31 c0                	xor    %eax,%eax
    1bea:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    1bf1:	48 05 20 00 00 00    	add    $0x20,%rax
    1bf7:	48 8b 00             	mov    (%rax),%rax
    1bfa:	50                   	push   %rax
    1bfb:	48 b8 06 00 00 00 00 	movabs $0x6,%rax
    1c02:	00 00 00 
    1c05:	50                   	push   %rax
    1c06:	48 8b bc 24 08 00 00 	mov    0x8(%rsp),%rdi
    1c0d:	00 
    1c0e:	48 8b b4 24 00 00 00 	mov    0x0(%rsp),%rsi
    1c15:	00 
    1c16:	e8 52 5c 00 00       	call   0x786d
    1c1b:	48 83 c4 10          	add    $0x10,%rsp
    1c1f:	31 c0                	xor    %eax,%eax
    1c21:	31 c0                	xor    %eax,%eax
    1c23:	48 81 c4 a0 01 00 00 	add    $0x1a0,%rsp
    1c2a:	5d                   	pop    %rbp
    1c2b:	c3                   	ret

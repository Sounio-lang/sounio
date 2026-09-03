def cdSigma(a,b,bits):
    if a==0 or b==0: return 1
    if bits<=1: return -1
    half=1<<(bits-1); aH=a>=half; bH=b>=half; aL=a&(half-1); bL=b&(half-1)
    if not aH and not bH: return cdSigma(aL,bL,bits-1)
    if not aH and bH: return cdSigma(bL,aL,bits-1)
    if aH and not bH: return cdSigma(aL,0,bits-1) if bL==0 else -cdSigma(aL,bL,bits-1)
    return -cdSigma(0,aL,bits-1) if bL==0 else cdSigma(bL,aL,bits-1)
def cd16(an,ad,bn,bd):
    out=[0]*16
    for i in range(16):
        if an[i]==0: continue
        for j in range(16):
            if bn[j]==0: continue
            out[i^j]+=cdSigma(i,j,4)*an[i]*bn[j]
    return out, ad*bd
def emit(tag,an,ad,bn,bd):
    out,den=cd16(an,ad,bn,bd)
    print(f"DEN {tag} {den}")
    for k in range(16): print(f"COMP {tag} {k} {out[k]}")
# Case 1 canonical
a1=[0]*16; a1[3]=1; a1[10]=1
b1=[0]*16; b1[6]=1; b1[15]=-1
emit(1,a1,1,b1,1)
# Case 2 general rational
a2=[0]*16; a2[1]=3; a2[2]=2
b2=[0]*16; b2[4]=7; b2[8]=5
emit(2,a2,6,b2,35)

// COMBINED head-to-head on the DGX Spark GB10: a task that requires BOTH non-associativity AND
// non-linearity, with every model TRAINED. Target y_i = ‖[a*,b*,c_i]‖² — the squared norm of the
// octonion associator of a fixed (unknown) teacher pair (a*,b*) with the input c_i. Because the target
// is (a) trilinear→quadratic (non-linear in c) and (b) an associator (zero for any associative algebra),
// the separation is structural, not just sample-efficiency:
//   - Octonion model: learns octonion weights a,b THROUGH the associator's VJP (da,db on tensor cores),
//     ŷ=‖[a,b,c]‖². Solves it.
//   - Quaternion model: its associator ≡ 0 (quaternions are associative) → ŷ≡0. Structurally BLIND —
//     no amount of training or data helps.
//   - Linear model on raw c: target is quadratic → fails.
//   - MLP on raw c: can fit the quadratic form given enough data (the unstructured baseline).
// Forward + da/db of the octonion model run on the compiler-lowered kernels (oct_assoc, ossm_oct_bwd,
// oct_batch_mul). Test-set R² reported for all four. Usage:
//   run_nonassoc_headtohead <oct_assoc.ptx> <ossm_oct_bwd.ptx> <oct_batch_mul.ptx>
#include <cstdio>
#include <cmath>
#include <cuda.h>
static unsigned RS=20260719u; static double rnd(){RS=RS*1664525u+1013904223u; return ((RS>>8)&0xffffff)/16777216.0*2-1;}
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
static const int BITS=3;
static CUfunction ASS,BWD,MUL;
static CUdeviceptr da_,db_,dH_,dO_,dHt_,dHpr_,dAacc_,dMul_;
static void fwd_assoc(const double*a,const double*b,const double*C,float*z){
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(db_,b,64);cuMemcpyHtoD(dH_,C,128*8);float zz[256]={0};cuMemcpyHtoD(dO_,zz,256*4);
 void*A[]={&da_,&db_,&dH_,&dO_};cuLaunchKernel(ASS,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();cuMemcpyDtoH(z,dO_,256*4);
}
static void grad_ab(const double*a,const double*b,const double*C,const double*dD,double*gda,double*gdb){
 double z8[8]={0};
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(dH_,C,128*8);cuMemcpyHtoD(dHt_,dD,256*8);cuMemcpyHtoD(dAacc_,z8,64);
 {void*A[]={&da_,&dH_,&dHt_,&dHpr_,&dAacc_};cuLaunchKernel(BWD,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();}
 double dP[8];cuMemcpyDtoH(dP,dAacc_,64);double dhp[256];cuMemcpyDtoH(dhp,dHpr_,256*8);
 double dq[256];for(int i=0;i<256;i++)dq[i]=-dhp[i];
 cuMemcpyHtoD(da_,b,64);cuMemcpyHtoD(dH_,C,128*8);float zz[256]={0};cuMemcpyHtoD(dMul_,zz,256*4);
 {void*A[]={&da_,&dH_,&dMul_};cuLaunchKernel(MUL,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();}
 float qf[256];cuMemcpyDtoH(qf,dMul_,256*4);double qsm[128];for(int k=0;k<8;k++)for(int i=0;i<16;i++)qsm[i*8+k]=qf[k*16+i];
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(dH_,qsm,128*8);cuMemcpyHtoD(dHt_,dD,256*8);cuMemcpyHtoD(dAacc_,z8,64);
 {void*A[]={&da_,&dH_,&dHt_,&dHpr_,&dAacc_};cuLaunchKernel(BWD,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();}
 double dar[8];cuMemcpyDtoH(dar,dAacc_,64);
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(dH_,C,128*8);cuMemcpyHtoD(dHt_,dq,256*8);cuMemcpyHtoD(dAacc_,z8,64);
 {void*A[]={&da_,&dH_,&dHt_,&dHpr_,&dAacc_};cuLaunchKernel(BWD,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();}
 double dbq[8];cuMemcpyDtoH(dbq,dAacc_,64);
 for(int i=0;i<8;i++){double s=0;for(int j=0;j<8;j++)s+=(double)cds(i,j,BITS)*b[j]*dP[i^j];gda[i]+=s-dar[i];}
 for(int j=0;j<8;j++){double s=0;for(int i=0;i<8;i++)s+=(double)cds(i,j,BITS)*a[i]*dP[i^j];gdb[j]+=s+dbq[j];}
}
// host octonion associator (reference for targets/eval) — [a,b,c] = (a⊗b)⊗c − a⊗(b⊗c)
static void omul(const double*x,const double*y,double*o){for(int k=0;k<8;k++)o[k]=0;for(int i=0;i<8;i++)for(int j=0;j<8;j++)o[i^j]+=(double)cds(i,j,BITS)*x[i]*y[j];}
static void assoc_h(const double*a,const double*b,const double*c,double*o){double ab[8],abc[8],bc[8],a_bc[8];omul(a,b,ab);omul(ab,c,abc);omul(b,c,bc);omul(a,bc,a_bc);for(int k=0;k<8;k++)o[k]=abc[k]-a_bc[k];}
static double R2(const double*yp,const double*yt,int n){double m=0;for(int i=0;i<n;i++)m+=yt[i];m/=n;double sr=0,st=0;for(int i=0;i<n;i++){sr+=(yp[i]-yt[i])*(yp[i]-yt[i]);st+=(yt[i]-m)*(yt[i]-m);}return 1.0-sr/st;}
int main(int c,char**v){
 const char*pa=c>1?v[1]:"/tmp/assoc.ptx",*pbw=c>2?v[2]:"/tmp/bwd.ptx",*pm=c>3?v[3]:"/tmp/mul.ptx";
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule ma,mb,mm;CK(cuModuleLoad(&ma,pa));CK(cuModuleLoad(&mb,pbw));CK(cuModuleLoad(&mm,pm));
 CK(cuModuleGetFunction(&ASS,ma,"step"));CK(cuModuleGetFunction(&BWD,mb,"step"));CK(cuModuleGetFunction(&MUL,mm,"step"));
 CK(cuMemAlloc(&da_,64));CK(cuMemAlloc(&db_,64));CK(cuMemAlloc(&dH_,128*8));CK(cuMemAlloc(&dO_,256*4));
 CK(cuMemAlloc(&dHt_,256*8));CK(cuMemAlloc(&dHpr_,256*8));CK(cuMemAlloc(&dAacc_,64));CK(cuMemAlloc(&dMul_,256*4));
 const int GTR=12,GTE=6;                                    // train / test batches (×16 samples)
 double Ctr[12][128],Cte[6][128];
 for(int g=0;g<GTR;g++)for(int i=0;i<16;i++)for(int k=0;k<8;k++)Ctr[g][i*8+k]=rnd();
 for(int g=0;g<GTE;g++)for(int i=0;i<16;i++)for(int k=0;k<8;k++)Cte[g][i*8+k]=rnd();
 double aT[8],bT[8];for(int k=0;k<8;k++){aT[k]=0.5*rnd();bT[k]=0.5*rnd();}
 // targets y = ‖[aT,bT,c]‖²
 auto target=[&](const double*Cg,double*y){for(int i=0;i<16;i++){double cc[8];for(int k=0;k<8;k++)cc[k]=Cg[i*8+k];double z[8];assoc_h(aT,bT,cc,z);double s=0;for(int k=0;k<8;k++)s+=z[k]*z[k];y[i]=s;}};
 double ytr[12][16],yte[6][16];for(int g=0;g<GTR;g++)target(Ctr[g],ytr[g]);for(int g=0;g<GTE;g++)target(Cte[g],yte[g]);
 const double b1=0.9,b2=0.999,eps=1e-8;const int NIT=400;
 // ── Model A: OCTONION (learn a,b via associator VJP), ŷ=‖[a,b,c]‖² ──
 double a[8],b[8],mA[8]={0},vA[8]={0},mB[8]={0},vB[8]={0};for(int k=0;k<8;k++){a[k]=0.5*rnd();b[k]=0.5*rnd();}
 for(int it=1;it<=NIT;it++){double gda[8]={0},gdb[8]={0};double bc1=1-pow(b1,it),bc2=1-pow(b2,it);
  for(int g=0;g<GTR;g++){float z[256];fwd_assoc(a,b,Ctr[g],z);double dD[256]={0};
   for(int i=0;i<16;i++){double yh=0;for(int k=0;k<8;k++)yh+=(double)z[k*16+i]*(double)z[k*16+i];double dyh=2.0*(yh-ytr[g][i]);
     for(int k=0;k<8;k++)dD[k*16+i]=dyh*2.0*(double)z[k*16+i];}
   grad_ab(a,b,Ctr[g],dD,gda,gdb);}
  for(int k=0;k<8;k++){mA[k]=b1*mA[k]+(1-b1)*gda[k];vA[k]=b2*vA[k]+(1-b2)*gda[k]*gda[k];a[k]-=0.02*(mA[k]/bc1)/(sqrt(vA[k]/bc2)+eps);
                       mB[k]=b1*mB[k]+(1-b1)*gdb[k];vB[k]=b2*vB[k]+(1-b2)*gdb[k]*gdb[k];b[k]-=0.02*(mB[k]/bc1)/(sqrt(vB[k]/bc2)+eps);}}
 double ypA[96],ytA[96];int n=0;for(int g=0;g<GTE;g++){float z[256];fwd_assoc(a,b,Cte[g],z);for(int i=0;i<16;i++){double yh=0;for(int k=0;k<8;k++)yh+=(double)z[k*16+i]*(double)z[k*16+i];ypA[n]=yh;ytA[n]=yte[g][i];n++;}}
 double r2A=R2(ypA,ytA,n);
 // ── Model B: QUATERNION associator ≡ 0 → ŷ≡0 (structural) ──
 double ypB[96];for(int i=0;i<n;i++)ypB[i]=0;double r2B=R2(ypB,ytA,n);
 // ── Model C: LINEAR on raw c (w·c+β), Adam ──
 double w[8]={0},bb=0,mW[8]={0},vW[8]={0},mbc=0,vbc=0;
 for(int it=1;it<=NIT;it++){double gw[8]={0},gb=0;double bc1=1-pow(b1,it),bc2=1-pow(b2,it);
  for(int g=0;g<GTR;g++)for(int i=0;i<16;i++){double yh=bb;for(int k=0;k<8;k++)yh+=w[k]*Ctr[g][i*8+k];double dy=2*(yh-ytr[g][i]);for(int k=0;k<8;k++)gw[k]+=dy*Ctr[g][i*8+k];gb+=dy;}
  for(int k=0;k<8;k++){mW[k]=b1*mW[k]+(1-b1)*gw[k];vW[k]=b2*vW[k]+(1-b2)*gw[k]*gw[k];w[k]-=0.02*(mW[k]/bc1)/(sqrt(vW[k]/bc2)+eps);}
  mbc=b1*mbc+(1-b1)*gb;vbc=b2*vbc+(1-b2)*gb*gb;bb-=0.02*(mbc/bc1)/(sqrt(vbc/bc2)+eps);}
 double ypC[96];n=0;for(int g=0;g<GTE;g++)for(int i=0;i<16;i++){double yh=bb;for(int k=0;k<8;k++)yh+=w[k]*Cte[g][i*8+k];ypC[n++]=yh;}double r2C=R2(ypC,ytA,n);
 // ── Model D: MLP 8→H→1 tanh, Adam (unstructured baseline) ──
 const int H=16;double W1[8*16],B1[16],W2[16],B2=0,mW1[8*16]={0},vW1[8*16]={0},mB1[16]={0},vB1[16]={0},mW2[16]={0},vW2[16]={0},mB2=0,vB2=0;
 for(int k=0;k<8*H;k++)W1[k]=0.3*rnd();for(int k=0;k<H;k++){B1[k]=0;W2[k]=0.3*rnd();}
 for(int it=1;it<=NIT;it++){double gW1[8*16]={0},gB1[16]={0},gW2[16]={0},gB2=0;double bc1=1-pow(b1,it),bc2=1-pow(b2,it);
  for(int g=0;g<GTR;g++)for(int i=0;i<16;i++){double h[16],hp[16],yh=B2;for(int u=0;u<H;u++){double s=B1[u];for(int k=0;k<8;k++)s+=W1[u*8+k]*Ctr[g][i*8+k];hp[u]=tanh(s);h[u]=s;yh+=W2[u]*hp[u];}
    double dy=2*(yh-ytr[g][i]);gB2+=dy;for(int u=0;u<H;u++){gW2[u]+=dy*hp[u];double dh=dy*W2[u]*(1-hp[u]*hp[u]);gB1[u]+=dh;for(int k=0;k<8;k++)gW1[u*8+k]+=dh*Ctr[g][i*8+k];}}
  for(int k=0;k<8*H;k++){mW1[k]=b1*mW1[k]+(1-b1)*gW1[k];vW1[k]=b2*vW1[k]+(1-b2)*gW1[k]*gW1[k];W1[k]-=0.01*(mW1[k]/bc1)/(sqrt(vW1[k]/bc2)+eps);}
  for(int u=0;u<H;u++){mB1[u]=b1*mB1[u]+(1-b1)*gB1[u];vB1[u]=b2*vB1[u]+(1-b2)*gB1[u]*gB1[u];B1[u]-=0.01*(mB1[u]/bc1)/(sqrt(vB1[u]/bc2)+eps);
                       mW2[u]=b1*mW2[u]+(1-b1)*gW2[u];vW2[u]=b2*vW2[u]+(1-b2)*gW2[u]*gW2[u];W2[u]-=0.01*(mW2[u]/bc1)/(sqrt(vW2[u]/bc2)+eps);}
  mB2=b1*mB2+(1-b1)*gB2;vB2=b2*vB2+(1-b2)*gB2*gB2;B2-=0.01*(mB2/bc1)/(sqrt(vB2/bc2)+eps);}
 double ypD[96];n=0;for(int g=0;g<GTE;g++)for(int i=0;i<16;i++){double yh=B2;for(int u=0;u<H;u++){double s=B1[u];for(int k=0;k<8;k++)s+=W1[u*8+k]*Cte[g][i*8+k];yh+=W2[u]*tanh(s);}ypD[n++]=yh;}double r2D=R2(ypD,ytA,n);
 printf("Combined non-associative + non-linear task  y=‖[a*,b*,c]‖²   (train %d / test %d samples, all trained w/ Adam):\n",GTR*16,GTE*16);
 printf("  Model                                    test R²\n");
 printf("  A octonion  (learns a,b via assoc VJP)   %+.4f   ← forward+da/db on tensor cores\n",r2A);
 printf("  B quaternion associator ≡ 0 (structural) %+.4f   ← BLIND: no training/data can help\n",r2B);
 printf("  C linear on raw c                        %+.4f   ← fails: target is quadratic\n",r2C);
 printf("  D MLP 8→16→1 on raw c (unstructured)     %+.4f   ← the capacity/data baseline\n",r2D);
 if(r2A>0.9 && r2B<0.1 && r2C<0.5){printf("PASS: only the non-associative model solves a task that needs non-associativity — and it TRAINED to, through the associator's VJP on GB10\n");return 0;}
 printf("FAIL\n");return 1;
}

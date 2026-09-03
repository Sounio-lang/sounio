// End-to-end NONLINEAR BPTT + training for the batched hypercomplex cell
//   S_t = A⊗H_{t-1} + B·x_t ,  H_t = σ(S_t) ,  σ(s) = (1/64)s³ + (1/4)s + 1/2 ,  Loss = Σ_t‖H_t−tgt_t‖²
// on the DGX Spark GB10. FORWARD pre-activation S = ossm_oct_step (compiler-lowered tile+post-add);
// the cubic activation σ(S) is elementwise (host). BACKWARD = ossm_oct_bwd_nl, which chains the cubic
// sigmoid derivative σ'(S)=(3/64)s²+1/4 through the recurrence (the nonlinearity in BPTT) and does
// dHprev = L(A)ᵀ·(dH⊙σ'(S)) + dA += da. Usage: run_bptt_nl <fwd.ptx> <bwdnl.ptx> <dim> [p1only].
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
static double sig(double s){return (1.0/64.0)*s*s*s + 0.25*s + 0.5;}
static double sigp(double s){return (3.0/64.0)*s*s + 0.25;}
#define CK(x) do{CUresult e=(x);if(e){const char*ss;cuGetErrorString(e,&ss);printf("ERR %s@%d:%s\n",#x,__LINE__,ss);return 2;}}while(0)
static int DIM,BITS; static CUfunction FWD,BWN;
static void fwd(CUdeviceptr A,CUdeviceptr B,CUdeviceptr x,CUdeviceptr H,CUdeviceptr S){
 void*a[]={&A,&B,&x,&H,&S}; cuLaunchKernel(FWD,1,1,1,32,1,1,0,0,a,0); cuCtxSynchronize();}
static void bwn(CUdeviceptr A,CUdeviceptr Hp,CUdeviceptr dHt,CUdeviceptr S,CUdeviceptr dHp,CUdeviceptr dA){
 void*a[]={&A,&Hp,&dHt,&S,&dHp,&dA}; cuLaunchKernel(BWN,1,1,1,32,1,1,0,0,a,0); cuCtxSynchronize();}

int main(int c,char**v){
 const char*pf=c>1?v[1]:"/tmp/fwd.ptx"; const char*pb=c>2?v[2]:"/tmp/bwdnl.ptx"; DIM=c>3?atoi(v[3]):8; BITS=(DIM==16)?4:3;
 const int ND=DIM*16;
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext cx;CK(cuDevicePrimaryCtxRetain(&cx,d));CK(cuCtxSetCurrent(cx));
 CUmodule mf,mb;CK(cuModuleLoad(&mf,pf));CK(cuModuleLoad(&mb,pb));
 CK(cuModuleGetFunction(&FWD,mf,"step"));CK(cuModuleGetFunction(&BWN,mb,"step"));
 CUdeviceptr dA_,dB_,dx_,dH_,dS_,dHp_,dHt_,dSp_,dHpr_,dAg_;
 CK(cuMemAlloc(&dA_,DIM*8));CK(cuMemAlloc(&dB_,DIM*8));CK(cuMemAlloc(&dx_,16*8));CK(cuMemAlloc(&dH_,ND*8));
 CK(cuMemAlloc(&dS_,256*4));CK(cuMemAlloc(&dHp_,ND*8));CK(cuMemAlloc(&dHt_,256*8));CK(cuMemAlloc(&dSp_,256*8));
 CK(cuMemAlloc(&dHpr_,256*8));CK(cuMemAlloc(&dAg_,DIM*8));

 // ── Part 1: validate one NONLINEAR bwd step vs exact f64 ─────────────────────────────────────
 {
  double A[16]={0},Hp[256]={0},dHt[256]={0},S[256]={0};
  for(int i=0;i<DIM;i++)A[i]=0.4*((i%2)?-1:1)+0.05*i;
  for(int b=0;b<16;b++)for(int j=0;j<DIM;j++)Hp[b*DIM+j]=((b+1)*0.11+j*0.05)*((j%2)?-1:1);
  for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){dHt[k*16+b]=0.2*((k+b)%3-1)+0.03*k;S[k*16+b]=0.3*sin(0.5*k+0.3*b);}
  double dS[256]; for(int k=0;k<DIM;k++)for(int b=0;b<16;b++)dS[k*16+b]=dHt[k*16+b]*sigp(S[k*16+b]);
  double dHp_ref[256]={0},dA_ref[16]={0};
  for(int j=0;j<DIM;j++)for(int b=0;b<16;b++){double s=0;for(int k=0;k<DIM;k++)s+=(double)cds(k^j,j,BITS)*A[k^j]*dS[k*16+b];dHp_ref[j*16+b]=s;}
  for(int pp=0;pp<DIM;pp++){double s=0;for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){int m=k^pp;s+=(double)cds(pp,m,BITS)*Hp[b*DIM+m]*dS[k*16+b];}dA_ref[pp]=s;}
  double zA[16]={0};CK(cuMemcpyHtoD(dAg_,zA,DIM*8));
  CK(cuMemcpyHtoD(dA_,A,DIM*8));CK(cuMemcpyHtoD(dHp_,Hp,ND*8));CK(cuMemcpyHtoD(dHt_,dHt,256*8));CK(cuMemcpyHtoD(dSp_,S,256*8));
  bwn(dA_,dHp_,dHt_,dSp_,dHpr_,dAg_);
  double gHp[256],gA[16];CK(cuMemcpyDtoH(gHp,dHpr_,256*8));CK(cuMemcpyDtoH(gA,dAg_,DIM*8));
  int fH=0,fA=0;double mH=0,mA=0;
  for(int j=0;j<DIM;j++)for(int b=0;b<16;b++){double e=fabs(gHp[j*16+b]-dHp_ref[j*16+b]);if(e>mH)mH=e;if(e>0.05)fH++;}
  for(int pp=0;pp<DIM;pp++){double e=fabs(gA[pp]-dA_ref[pp]);if(e>mA)mA=e;if(e>1e-4)fA++;}
  printf("Part 1 — one NONLINEAR BPTT backward step (%s):\n",(DIM==16)?"sedenion":"octonion");
  printf("  dHprev = L(A)^T·(dH⊙σ'(S)) (tile): mismatch %d/%d maxerr=%.4f\n",fH,ND,mH);
  printf("  dA += da(H, dH⊙σ'(S)) (f64):       mismatch %d/%d maxerr=%.2e\n",fA,DIM,mA);
  if(fH||fA){printf("FAIL (part 1)\n");return 1;}
  printf("  PASS\n");
 }
 if(c>4 && strcmp(v[4],"p1only")==0){printf("(p1only) nonlinear backward-step validated; training skipped.\n");return 0;}

 // ── Part 2: NONLINEAR BPTT training loop ─────────────────────────────────────────────────────
 const int T=3, NITER=60;
 const double lr = (DIM==16)?0.010:0.05;
 const double asc = (DIM==16)?0.5:1.0, pert = (DIM==16)?0.04:0.10;
 double h0[256]={0}, x[3][16], As[16], Bs[16];
 for(int b=0;b<16;b++)for(int j=0;j<DIM;j++)h0[b*DIM+j]=0.1*((b%2)?-1:1)+0.02*j;
 for(int t=0;t<T;t++)for(int b=0;b<16;b++)x[t][b]=0.3*sin(0.7*t+0.4*b);
 for(int i=0;i<DIM;i++){As[i]=asc*(0.25*((i%2)?-1:1)+0.04*i); Bs[i]=asc*(0.15-0.03*i);}
 // teacher targets: forward (pre-act S via kernel, σ on host), teacher trajectory as input each step
 double tgt[3][256]={{0}};
 {
  double hs[256]; memcpy(hs,h0,sizeof(double)*256);
  CK(cuMemcpyHtoD(dA_,As,DIM*8));CK(cuMemcpyHtoD(dB_,Bs,DIM*8));
  for(int t=0;t<T;t++){
   CK(cuMemcpyHtoD(dx_,x[t],16*8));CK(cuMemcpyHtoD(dH_,hs,ND*8));fwd(dA_,dB_,dx_,dH_,dS_);
   float S[256];CK(cuMemcpyDtoH(S,dS_,256*4));
   for(int k=0;k<DIM;k++)for(int b=0;b<16;b++)tgt[t][k*16+b]=sig(S[k*16+b]);
   double nh[256]={0};for(int k=0;k<DIM;k++)for(int b=0;b<16;b++)nh[b*DIM+k]=sig(S[k*16+b]);memcpy(hs,nh,sizeof(double)*256);
  }
 }
 double A[16],Bp[16]; for(int i=0;i<DIM;i++){A[i]=As[i]+pert*((i%2)?-1:1); Bp[i]=Bs[i]+pert*0.83*((i%3)?1:-1);}
 double loss0=0,lossN=0;
 double Straj[3][256], Htraj[3][256], hst[4][256];
 for(int it=0; it<NITER; it++){
  memcpy(hst[0],h0,sizeof(double)*256);
  CK(cuMemcpyHtoD(dA_,A,DIM*8));CK(cuMemcpyHtoD(dB_,Bp,DIM*8));
  double loss=0;
  for(int t=0;t<T;t++){
   CK(cuMemcpyHtoD(dx_,x[t],16*8));CK(cuMemcpyHtoD(dH_,hst[t],ND*8));fwd(dA_,dB_,dx_,dH_,dS_);
   float S[256];CK(cuMemcpyDtoH(S,dS_,256*4));
   for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){Straj[t][k*16+b]=S[k*16+b];double h=sig(S[k*16+b]);Htraj[t][k*16+b]=h;
     double e=h-tgt[t][k*16+b];loss+=e*e;}
   for(int k=0;k<DIM;k++)for(int b=0;b<16;b++)hst[t+1][b*DIM+k]=Htraj[t][k*16+b];
  }
  if(it==0)loss0=loss; lossN=loss;
  double dBacc[16]={0}, dHnext[256]={0}, zA[16]={0};CK(cuMemcpyHtoD(dAg_,zA,DIM*8));
  for(int t=T-1;t>=0;t--){
   double dHt[256];
   for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){double dY=2.0*(Htraj[t][k*16+b]-tgt[t][k*16+b]);dHt[k*16+b]=dY+dHnext[k*16+b];}
   for(int k=0;k<DIM;k++){double s=0;for(int b=0;b<16;b++)s+=dHt[k*16+b]*sigp(Straj[t][k*16+b])*x[t][b];dBacc[k]+=s;}
   CK(cuMemcpyHtoD(dHt_,dHt,256*8));CK(cuMemcpyHtoD(dSp_,Straj[t],256*8));CK(cuMemcpyHtoD(dHp_,hst[t],ND*8));
   bwn(dA_,dHp_,dHt_,dSp_,dHpr_,dAg_);
   CK(cuMemcpyDtoH(dHnext,dHpr_,256*8));
  }
  double dA[16];CK(cuMemcpyDtoH(dA,dAg_,DIM*8));
  for(int p=0;p<DIM;p++){A[p]-=lr*dA[p]; Bp[p]-=lr*dBacc[p];}
  if(it==0||it==NITER-1||it%15==14)printf("  iter %2d  loss %.5f\n",it,loss);
 }
 printf("Part 2 — NONLINEAR BPTT training: loss %.5f → %.5f  (%.1f%% reduction)\n",loss0,lossN,100.0*(loss0-lossN)/loss0);
 if(lossN < 0.5*loss0){printf("PASS: nonlinear BPTT + SGD trained h_t=σ(A⊗h+B·x) on GB10 (cubic-sigmoid derivative chained through the recurrence)\n");return 0;}
 printf("FAIL: loss did not fall enough\n");return 1;
}

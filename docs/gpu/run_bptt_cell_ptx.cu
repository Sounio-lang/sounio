// End-to-end training of the FULL O-SSM cell with the real readout, on the DGX Spark GB10:
//   S_t = A⊗H_{t-1} + B·x_t ,  H_t = σ(S_t) ,  y_t = Re(C⊗H_t) ,  Loss = Σ_t Σ_b (y_t[b] − tgt_t[b])²
// Four compiler-lowered kernels: ossm_oct_step (S), ossm_oct_readout (y=Re(C⊗H)), ossm_oct_bwd_nl
// (dA via σ'), and ossm_oct_bwd applied to (C, H, row-0 gradient) for the readout backward (dC + the
// activation-space gradient dH_readout = L(C)ᵀ·dD_row0). Trains A, B AND C. Recurrence unroll + SGD
// on the host. Usage: run_bptt_cell <step.ptx> <readout.ptx> <bwd.ptx> <bwdnl.ptx> <dim> [p1only].
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
static double sig(double s){return (1.0/64.0)*s*s*s+0.25*s+0.5;}
static double sigp(double s){return (3.0/64.0)*s*s+0.25;}
#define CK(x) do{CUresult e=(x);if(e){const char*ss;cuGetErrorString(e,&ss);printf("ERR %s@%d:%s\n",#x,__LINE__,ss);return 2;}}while(0)
static int DIM,BITS; static CUfunction FWD,RDO,BWD,BWN;
static void fwd(CUdeviceptr A,CUdeviceptr B,CUdeviceptr x,CUdeviceptr H,CUdeviceptr S){void*a[]={&A,&B,&x,&H,&S};cuLaunchKernel(FWD,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void rdo(CUdeviceptr C,CUdeviceptr H,CUdeviceptr y){void*a[]={&C,&H,&y};cuLaunchKernel(RDO,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void bwd(CUdeviceptr C,CUdeviceptr Hp,CUdeviceptr dD,CUdeviceptr dHp,CUdeviceptr dC){void*a[]={&C,&Hp,&dD,&dHp,&dC};cuLaunchKernel(BWD,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void bwn(CUdeviceptr A,CUdeviceptr Hp,CUdeviceptr dHt,CUdeviceptr S,CUdeviceptr dHp,CUdeviceptr dA){void*a[]={&A,&Hp,&dHt,&S,&dHp,&dA};cuLaunchKernel(BWN,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}

int main(int c,char**v){
 const char*ps=c>1?v[1]:"/tmp/step.ptx"; const char*pr=c>2?v[2]:"/tmp/rdo.ptx";
 const char*pb=c>3?v[3]:"/tmp/bwd.ptx"; const char*pn=c>4?v[4]:"/tmp/bwdnl.ptx"; DIM=c>5?atoi(v[5]):8; BITS=(DIM==16)?4:3;
 const int ND=DIM*16;
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext cx;CK(cuDevicePrimaryCtxRetain(&cx,d));CK(cuCtxSetCurrent(cx));
 CUmodule ms,mr,mb,mn;CK(cuModuleLoad(&ms,ps));CK(cuModuleLoad(&mr,pr));CK(cuModuleLoad(&mb,pb));CK(cuModuleLoad(&mn,pn));
 CK(cuModuleGetFunction(&FWD,ms,"step"));CK(cuModuleGetFunction(&RDO,mr,"step"));CK(cuModuleGetFunction(&BWD,mb,"step"));CK(cuModuleGetFunction(&BWN,mn,"step"));
 CUdeviceptr dA_,dB_,dC_,dx_,dHsm_,dHd_,dS_,dy_,dD_,dHp_,dHt_,dSp_,dgA_,dgC_;
 CK(cuMemAlloc(&dA_,DIM*8));CK(cuMemAlloc(&dB_,DIM*8));CK(cuMemAlloc(&dC_,DIM*8));CK(cuMemAlloc(&dx_,16*8));
 CK(cuMemAlloc(&dHsm_,ND*8));CK(cuMemAlloc(&dHd_,256*8));CK(cuMemAlloc(&dS_,256*4));CK(cuMemAlloc(&dy_,16*8));
 CK(cuMemAlloc(&dD_,256*8));CK(cuMemAlloc(&dHp_,256*8));CK(cuMemAlloc(&dHt_,256*8));CK(cuMemAlloc(&dSp_,256*8));
 CK(cuMemAlloc(&dgA_,DIM*8));CK(cuMemAlloc(&dgC_,DIM*8));

 // ── Part 1: validate the real readout y = Re(C⊗H) vs host ────────────────────────────────────
 {
  double C[16]={0},Hd[256]={0};
  for(int i=0;i<DIM;i++)C[i]=0.3*((i%2)?-1:1)+0.04*i;
  for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)Hd[p*16+b]=0.2*sin(0.4*p+0.6*b);
  double yref[16]={0};
  for(int b=0;b<16;b++){double s=0;for(int p=0;p<DIM;p++)s+=(double)cds(p,p,BITS)*C[p]*Hd[p*16+b];yref[b]=s;}
  CK(cuMemcpyHtoD(dC_,C,DIM*8));CK(cuMemcpyHtoD(dHd_,Hd,256*8));
  rdo(dC_,dHd_,dy_);
  double gy[16];CK(cuMemcpyDtoH(gy,dy_,16*8));
  int f=0;double mx=0;for(int b=0;b<16;b++){double e=fabs(gy[b]-yref[b]);if(e>mx)mx=e;if(e>1e-9)f++;}
  printf("Part 1 — real readout y = Re(C⊗H) (%s): mismatch %d/16 maxerr=%.2e\n",(DIM==16)?"sedenion":"octonion",f,mx);
  if(f){printf("FAIL (part 1)\n");return 1;} printf("  PASS\n");
 }
 if(c>6 && strcmp(v[6],"p1only")==0){printf("(p1only) readout validated; training skipped.\n");return 0;}

 // ── Part 2: full-cell training (learns A, B, C) ──────────────────────────────────────────────
 const int T=3, NITER=120;
 const double lr=(DIM==16)?0.003:0.012;
 const double asc=(DIM==16)?0.5:1.0, pert=(DIM==16)?0.035:0.07;
 double h0[256]={0}, x[3][16], As[16],Bs[16],Cs[16];
 for(int b=0;b<16;b++)for(int j=0;j<DIM;j++)h0[b*DIM+j]=0.1*((b%2)?-1:1)+0.02*j;
 for(int t=0;t<T;t++)for(int b=0;b<16;b++)x[t][b]=0.3*sin(0.7*t+0.4*b);
 for(int i=0;i<DIM;i++){As[i]=asc*(0.25*((i%2)?-1:1)+0.04*i);Bs[i]=asc*(0.15-0.03*i);Cs[i]=0.3*((i%2)?1:-1)+0.03*i;}
 // teacher targets y_t via the full forward (step+σ+readout)
 double tgt[3][16]={{0}};
 {
  double hsm[256];memcpy(hsm,h0,sizeof(double)*256);
  CK(cuMemcpyHtoD(dA_,As,DIM*8));CK(cuMemcpyHtoD(dB_,Bs,DIM*8));CK(cuMemcpyHtoD(dC_,Cs,DIM*8));
  for(int t=0;t<T;t++){
   CK(cuMemcpyHtoD(dx_,x[t],16*8));CK(cuMemcpyHtoD(dHsm_,hsm,ND*8));fwd(dA_,dB_,dx_,dHsm_,dS_);
   float S[256];CK(cuMemcpyDtoH(S,dS_,256*4)); double hd[256];
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hd[p*16+b]=sig(S[p*16+b]);
   CK(cuMemcpyHtoD(dHd_,hd,256*8));rdo(dC_,dHd_,dy_);CK(cuMemcpyDtoH(tgt[t],dy_,16*8));
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hsm[b*DIM+p]=hd[p*16+b];
  }
 }
 double A[16],Bp[16],C[16];for(int i=0;i<DIM;i++){A[i]=As[i]+pert*((i%2)?-1:1);Bp[i]=Bs[i]+pert*0.8*((i%3)?1:-1);C[i]=Cs[i]+pert*0.9*((i%2)?-1:1);}
 double loss0=0,lossN=0;
 double Straj[3][256],Hd[3][256],hsm[4][256];
 for(int it=0; it<NITER; it++){
  memcpy(hsm[0],h0,sizeof(double)*256);
  CK(cuMemcpyHtoD(dA_,A,DIM*8));CK(cuMemcpyHtoD(dB_,Bp,DIM*8));CK(cuMemcpyHtoD(dC_,C,DIM*8));
  double loss=0, ytr[3][16];
  for(int t=0;t<T;t++){
   CK(cuMemcpyHtoD(dx_,x[t],16*8));CK(cuMemcpyHtoD(dHsm_,hsm[t],ND*8));fwd(dA_,dB_,dx_,dHsm_,dS_);
   float S[256];CK(cuMemcpyDtoH(S,dS_,256*4));
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++){Straj[t][p*16+b]=S[p*16+b];Hd[t][p*16+b]=sig(S[p*16+b]);}
   CK(cuMemcpyHtoD(dHd_,Hd[t],256*8));rdo(dC_,dHd_,dy_);CK(cuMemcpyDtoH(ytr[t],dy_,16*8));
   for(int b=0;b<16;b++){double e=ytr[t][b]-tgt[t][b];loss+=e*e;}
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hsm[t+1][b*DIM+p]=Hd[t][p*16+b];
  }
  if(it==0)loss0=loss; lossN=loss;
  // backward
  double dBacc[16]={0}, dHnext[256]={0}, z[16]={0};
  CK(cuMemcpyHtoD(dgA_,z,DIM*8));CK(cuMemcpyHtoD(dgC_,z,DIM*8));
  for(int t=T-1;t>=0;t--){
   // readout backward: dD_row0[k*16+b] = (k==0? dy:0); ossm_oct_bwd(C, H_t_sm, dD_row0) → dH_readout, dC
   double dD[256]={0}; for(int b=0;b<16;b++)dD[0*16+b]=2.0*(ytr[t][b]-tgt[t][b]);
   double Hsm[256]={0}; for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)Hsm[b*DIM+p]=Hd[t][p*16+b];
   CK(cuMemcpyHtoD(dHp_,dD,256*8)); // dD_row0 (reuse dHp_ as dD input to bwd)
   CK(cuMemcpyHtoD(dSp_,Hsm,ND*8)); // H_t state-major (Hprev of the readout multiply)
   bwd(dC_,dSp_,dHp_,dD_,dgC_);      // dC_=C, dSp_=H, dHp_=dD_row0 → dD_=dH_readout(Dlayout), dgC_+=dC
   double dHr[256];CK(cuMemcpyDtoH(dHr,dD_,256*8));
   // total activation gradient dH_t = dH_readout + dH_recurrent
   double dHt[256]; for(int k=0;k<DIM;k++)for(int b=0;b<16;b++)dHt[k*16+b]=dHr[k*16+b]+dHnext[k*16+b];
   // dB += Σ (dHt·σ'(S))·x
   for(int k=0;k<DIM;k++){double s=0;for(int b=0;b<16;b++)s+=dHt[k*16+b]*sigp(Straj[t][k*16+b])*x[t][b];dBacc[k]+=s;}
   // recurrent backward (nonlinear): ossm_oct_bwd_nl(A, h_{t-1}_sm, dHt, S_t) → dH_recurrent, dA
   CK(cuMemcpyHtoD(dHt_,dHt,256*8));CK(cuMemcpyHtoD(dSp_,Straj[t],256*8));CK(cuMemcpyHtoD(dHp_,hsm[t],ND*8));
   bwn(dA_,dHp_,dHt_,dSp_,dD_,dgA_);
   CK(cuMemcpyDtoH(dHnext,dD_,256*8));
  }
  double dA[16],dC[16];CK(cuMemcpyDtoH(dA,dgA_,DIM*8));CK(cuMemcpyDtoH(dC,dgC_,DIM*8));
  for(int p=0;p<DIM;p++){A[p]-=lr*dA[p];Bp[p]-=lr*dBacc[p];C[p]-=lr*dC[p];}
  if(it==0||it==NITER-1||it%20==19)printf("  iter %2d  loss %.5f\n",it,loss);
 }
 printf("Part 2 — FULL-CELL training (A,B,C via Re(C⊗σ(A⊗h+B·x))): loss %.5f → %.5f  (%.1f%% reduction)\n",loss0,lossN,100.0*(loss0-lossN)/loss0);
 if(lossN < 0.5*loss0){printf("PASS: trained the full O-SSM cell (state recurrence + nonlinearity + real readout) on GB10\n");return 0;}
 printf("FAIL: loss did not fall enough\n");return 1;
}

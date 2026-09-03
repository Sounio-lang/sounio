// Adam optimizer for the FULL O-SSM cell training on the DGX Spark GB10 — robust convergence.
//   S_t = A⊗H_{t-1} + B·x_t ,  H_t = σ(S_t) ,  y_t = Re(C⊗H_t) ,  Loss = Σ_t Σ_b (y_t[b] − tgt_t[b])²
// The forward/backward passes are the SAME four compiler-lowered tensor-core kernels as the SGD cell
// harness (ossm_oct_step / ossm_oct_readout / ossm_oct_bwd / ossm_oct_bwd_nl); only the parameter
// update changes. A/B/C are tiny (dim≤16 values each) so the optimizer is host-side; the heavy math
// stays in the Sounio kernels. Adam (β1=0.9, β2=0.999, ε=1e-8) converges with a SINGLE lr for both
// octonion and sedenion — no per-dim lr tuning — and reaches a lower loss than plain SGD.
// Usage: run_bptt_adam <step.ptx> <readout.ptx> <bwd.ptx> <bwdnl.ptx> <dim>
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
static int DIM,BITS,ND;
static CUfunction FWD,RDO,BWD,BWN;
static void fwd(CUdeviceptr A,CUdeviceptr B,CUdeviceptr x,CUdeviceptr H,CUdeviceptr S){void*a[]={&A,&B,&x,&H,&S};cuLaunchKernel(FWD,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void rdo(CUdeviceptr C,CUdeviceptr H,CUdeviceptr y){void*a[]={&C,&H,&y};cuLaunchKernel(RDO,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void bwd(CUdeviceptr C,CUdeviceptr Hp,CUdeviceptr dD,CUdeviceptr dHp,CUdeviceptr dC){void*a[]={&C,&Hp,&dD,&dHp,&dC};cuLaunchKernel(BWD,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void bwn(CUdeviceptr A,CUdeviceptr Hp,CUdeviceptr dHt,CUdeviceptr S,CUdeviceptr dHp,CUdeviceptr dA){void*a[]={&A,&Hp,&dHt,&S,&dHp,&dA};cuLaunchKernel(BWN,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
// device scratch (globals)
static CUdeviceptr dA_,dB_,dC_,dx_,dHsm_,dHd_,dS_,dy_,dD_,dHp_,dHt_,dSp_,dgA_,dgC_;
// problem data (globals)
static int T; static double h0[256],x[3][16],tgt[3][16];

// one training iteration: forward+loss into *loss, gradients into gA,gB,gC.
static int iterate(double*A,double*Bp,double*C,double*loss,double*gA,double*gB,double*gC){
 double hsm[4][256],Straj[3][256],Hd[3][256],ytr[3][16];
 memcpy(hsm[0],h0,sizeof(double)*256);
 CK(cuMemcpyHtoD(dA_,A,DIM*8));CK(cuMemcpyHtoD(dB_,Bp,DIM*8));CK(cuMemcpyHtoD(dC_,C,DIM*8));
 double L=0;
 for(int t=0;t<T;t++){
  CK(cuMemcpyHtoD(dx_,x[t],16*8));CK(cuMemcpyHtoD(dHsm_,hsm[t],ND*8));fwd(dA_,dB_,dx_,dHsm_,dS_);
  float S[256];CK(cuMemcpyDtoH(S,dS_,256*4));
  for(int p=0;p<DIM;p++)for(int b=0;b<16;b++){Straj[t][p*16+b]=S[p*16+b];Hd[t][p*16+b]=sig(S[p*16+b]);}
  CK(cuMemcpyHtoD(dHd_,Hd[t],256*8));rdo(dC_,dHd_,dy_);CK(cuMemcpyDtoH(ytr[t],dy_,16*8));
  for(int b=0;b<16;b++){double e=ytr[t][b]-tgt[t][b];L+=e*e;}
  for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hsm[t+1][b*DIM+p]=Hd[t][p*16+b];
 }
 *loss=L;
 double dBacc[16]={0},dHnext[256]={0},z[16]={0};
 CK(cuMemcpyHtoD(dgA_,z,DIM*8));CK(cuMemcpyHtoD(dgC_,z,DIM*8));
 for(int t=T-1;t>=0;t--){
  double dD[256]={0};for(int b=0;b<16;b++)dD[b]=2.0*(ytr[t][b]-tgt[t][b]);
  double Hsm[256]={0};for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)Hsm[b*DIM+p]=Hd[t][p*16+b];
  CK(cuMemcpyHtoD(dHp_,dD,256*8));CK(cuMemcpyHtoD(dSp_,Hsm,ND*8));
  bwd(dC_,dSp_,dHp_,dD_,dgC_);
  double dHr[256];CK(cuMemcpyDtoH(dHr,dD_,256*8));
  double dHt[256];for(int k=0;k<DIM;k++)for(int b=0;b<16;b++)dHt[k*16+b]=dHr[k*16+b]+dHnext[k*16+b];
  for(int k=0;k<DIM;k++){double s=0;for(int b=0;b<16;b++)s+=dHt[k*16+b]*sigp(Straj[t][k*16+b])*x[t][b];dBacc[k]+=s;}
  CK(cuMemcpyHtoD(dHt_,dHt,256*8));CK(cuMemcpyHtoD(dSp_,Straj[t],256*8));CK(cuMemcpyHtoD(dHp_,hsm[t],ND*8));
  bwn(dA_,dHp_,dHt_,dSp_,dD_,dgA_);
  CK(cuMemcpyDtoH(dHnext,dD_,256*8));
 }
 CK(cuMemcpyDtoH(gA,dgA_,DIM*8));CK(cuMemcpyDtoH(gC,dgC_,DIM*8));memcpy(gB,dBacc,sizeof(double)*16);
 return 0;
}
// full training run; use_adam=0 SGD, 1 Adam. returns final loss (via *out) and loss0.
static double train(int use_adam,double lr,double*A0,double*B0,double*C0,double*out,int NITER){
 double A[16],Bp[16],C[16];memcpy(A,A0,128);memcpy(Bp,B0,128);memcpy(C,C0,128);
 double mA[16]={0},vA[16]={0},mB[16]={0},vB[16]={0},mC[16]={0},vC[16]={0};
 const double b1=0.9,b2=0.999,eps=1e-8;
 double loss0=0,loss=0;
 for(int it=1; it<=NITER; it++){
  double gA[16],gB[16],gC[16];
  iterate(A,Bp,C,&loss,gA,gB,gC);
  if(it==1)loss0=loss;
  if(!use_adam){
   for(int p=0;p<DIM;p++){A[p]-=lr*gA[p];Bp[p]-=lr*gB[p];C[p]-=lr*gC[p];}
  }else{
   double bc1=1.0-pow(b1,it), bc2=1.0-pow(b2,it);
   for(int p=0;p<DIM;p++){
#define ADAM(TH,G,M,V) M[p]=b1*M[p]+(1-b1)*G[p]; V[p]=b2*V[p]+(1-b2)*G[p]*G[p]; TH[p]-=lr*(M[p]/bc1)/(sqrt(V[p]/bc2)+eps);
    ADAM(A,gA,mA,vA) ADAM(Bp,gB,mB,vB) ADAM(C,gC,mC,vC)
#undef ADAM
   }
  }
 }
 *out=loss; return loss0;
}

int main(int c,char**v){
 const char*ps=c>1?v[1]:"/tmp/step.ptx";const char*pr=c>2?v[2]:"/tmp/rdo.ptx";
 const char*pb=c>3?v[3]:"/tmp/bwd.ptx";const char*pn=c>4?v[4]:"/tmp/bwdnl.ptx";DIM=c>5?atoi(v[5]):8;BITS=(DIM==16)?4:3;ND=DIM*16;
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext cx;CK(cuDevicePrimaryCtxRetain(&cx,d));CK(cuCtxSetCurrent(cx));
 CUmodule ms,mr,mb,mn;CK(cuModuleLoad(&ms,ps));CK(cuModuleLoad(&mr,pr));CK(cuModuleLoad(&mb,pb));CK(cuModuleLoad(&mn,pn));
 CK(cuModuleGetFunction(&FWD,ms,"step"));CK(cuModuleGetFunction(&RDO,mr,"step"));CK(cuModuleGetFunction(&BWD,mb,"step"));CK(cuModuleGetFunction(&BWN,mn,"step"));
 CK(cuMemAlloc(&dA_,DIM*8));CK(cuMemAlloc(&dB_,DIM*8));CK(cuMemAlloc(&dC_,DIM*8));CK(cuMemAlloc(&dx_,16*8));
 CK(cuMemAlloc(&dHsm_,ND*8));CK(cuMemAlloc(&dHd_,256*8));CK(cuMemAlloc(&dS_,256*4));CK(cuMemAlloc(&dy_,16*8));
 CK(cuMemAlloc(&dD_,256*8));CK(cuMemAlloc(&dHp_,256*8));CK(cuMemAlloc(&dHt_,256*8));CK(cuMemAlloc(&dSp_,256*8));
 CK(cuMemAlloc(&dgA_,DIM*8));CK(cuMemAlloc(&dgC_,DIM*8));
 T=3;
 double asc=(DIM==16)?0.5:1.0, pert=(DIM==16)?0.035:0.07;
 double As[16],Bs[16],Cs[16];
 for(int b=0;b<16;b++)for(int j=0;j<DIM;j++)h0[b*DIM+j]=0.1*((b%2)?-1:1)+0.02*j;
 for(int t=0;t<T;t++)for(int b=0;b<16;b++)x[t][b]=0.3*sin(0.7*t+0.4*b);
 for(int i=0;i<DIM;i++){As[i]=asc*(0.25*((i%2)?-1:1)+0.04*i);Bs[i]=asc*(0.15-0.03*i);Cs[i]=0.3*((i%2)?1:-1)+0.03*i;}
 // teacher targets via the full forward
 {double hsm[256];memcpy(hsm,h0,sizeof(double)*256);
  CK(cuMemcpyHtoD(dA_,As,DIM*8));CK(cuMemcpyHtoD(dB_,Bs,DIM*8));CK(cuMemcpyHtoD(dC_,Cs,DIM*8));
  for(int t=0;t<T;t++){CK(cuMemcpyHtoD(dx_,x[t],16*8));CK(cuMemcpyHtoD(dHsm_,hsm,ND*8));fwd(dA_,dB_,dx_,dHsm_,dS_);
   float S[256];CK(cuMemcpyDtoH(S,dS_,256*4));double hd[256];
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hd[p*16+b]=sig(S[p*16+b]);
   CK(cuMemcpyHtoD(dHd_,hd,256*8));rdo(dC_,dHd_,dy_);CK(cuMemcpyDtoH(tgt[t],dy_,16*8));
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hsm[b*DIM+p]=hd[p*16+b];}}
 double A0[16],B0[16],C0[16];for(int i=0;i<DIM;i++){A0[i]=As[i]+pert*((i%2)?-1:1);B0[i]=Bs[i]+pert*0.8*((i%3)?1:-1);C0[i]=Cs[i]+pert*0.9*((i%2)?-1:1);}
 const char*nm=(DIM==16)?"sedenion":"octonion";
 const int NIT=200;
 // SGD at a single fixed lr (the same for both dims — deliberately NOT per-dim tuned) vs Adam.
 const double lr_fixed=0.01, lr_adam=0.01;
 double sgd_f,sgd0=train(0,lr_fixed,A0,B0,C0,&sgd_f,NIT);
 double adam_f,adam0=train(1,lr_adam,A0,B0,C0,&adam_f,NIT);
 int sgd_diverged = isnan(sgd_f) || isinf(sgd_f);
 printf("Full-cell training (%s), %d iters, single fixed lr=%.3f for BOTH algebras:\n",nm,NIT,lr_fixed);
 if(sgd_diverged) printf("  SGD : loss %.5f → DIVERGED (nan) — this fixed lr is too large for plain SGD\n",sgd0);
 else             printf("  SGD : loss %.5f → %.6f  (%.1f%%)\n",sgd0,sgd_f,100.0*(sgd0-sgd_f)/sgd0);
 printf("  Adam: loss %.5f → %.6f  (%.1f%%)\n",adam0,adam_f,100.0*(adam0-adam_f)/adam0);
 // Adam wins if it converged well AND either SGD diverged or Adam reached a lower loss.
 int ok = (adam_f < 0.05*adam0) && (sgd_diverged || adam_f < sgd_f);
 if(ok){printf("PASS: Adam converges robustly at a fixed lr where SGD %s — on GB10\n",
               sgd_diverged?"DIVERGES":"reaches a higher loss");return 0;}
 printf("FAIL: Adam did not converge enough\n");return 1;
}

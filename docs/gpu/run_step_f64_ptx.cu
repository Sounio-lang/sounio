// Validates the FULL-f64 O-SSM forward step on the DGX Spark GB10, against the f16→f32 tensor-core
// tile (ossm_oct_step). Part 1: S = A⊗H + B·x exactness vs a host f64 reference — the f64 kernel is
// exact, the f16 tile carries ~0.01 error. Part 2: a T=1 full-cell finite-difference gradient check —
// with the EXACT f64 forward the FD gradient matches the analytic f64 BPTT tightly, whereas the f16
// forward makes FD noisy (quantization). Usage: run_step_f64 <stepf64.ptx> <stepf16.ptx> <readout.ptx> <dim>.
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
static int DIM,BITS;
static CUfunction F64,F16,RDO;
static CUdeviceptr dA_,dB_,dx_,dH_,dSf,dS4,dC_,dHd_,dy_;
static void ref_step(const double*A,const double*B,const double*x,const double*Hsm,double*S){
 for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){double acc=B[k]*x[b];
   for(int j=0;j<DIM;j++)acc+=(double)cds(k^j,j,BITS)*A[k^j]*Hsm[b*DIM+j];
   S[k*16+b]=acc;}}
static void stepf64(CUdeviceptr A,CUdeviceptr B,CUdeviceptr x,CUdeviceptr H,CUdeviceptr S){void*a[]={&A,&B,&x,&H,&S};cuLaunchKernel(F64,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void stepf16(CUdeviceptr A,CUdeviceptr B,CUdeviceptr x,CUdeviceptr H,CUdeviceptr S){void*a[]={&A,&B,&x,&H,&S};cuLaunchKernel(F16,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
static void rdo(CUdeviceptr C,CUdeviceptr H,CUdeviceptr y){void*a[]={&C,&H,&y};cuLaunchKernel(RDO,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
// T=1 full-cell loss via a chosen forward step; σ + readout on host+kernel. y written to yout if !=0.
static double cell_loss(const double*A,const double*B,const double*x,const double*Hsm,const double*C,const double*tgt,int use_f64,double*yout){
 double y[16];
 if(cuMemcpyHtoD(dA_,A,DIM*8)||cuMemcpyHtoD(dB_,B,DIM*8)||cuMemcpyHtoD(dx_,x,16*8)||cuMemcpyHtoD(dH_,Hsm,DIM*16*8))return 0;
 double hd[256];
 if(use_f64){stepf64(dA_,dB_,dx_,dH_,dSf);double S[256];cuMemcpyDtoH(S,dSf,256*8);
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hd[p*16+b]=sig(S[p*16+b]);}
 else{stepf16(dA_,dB_,dx_,dH_,dS4);float S[256];cuMemcpyDtoH(S,dS4,256*4);
   for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)hd[p*16+b]=sig((double)S[p*16+b]);}
 cuMemcpyHtoD(dHd_,hd,256*8);rdo(dC_,dHd_,dy_);cuMemcpyDtoH(y,dy_,16*8);
 if(yout)memcpy(yout,y,16*8);
 double L=0;for(int b=0;b<16;b++){double e=y[b]-(tgt?tgt[b]:0.0);L+=e*e;} return L;
}

int main(int c,char**v){
 const char*p64=c>1?v[1]:"/tmp/f64.ptx";const char*p16=c>2?v[2]:"/tmp/f16.ptx";const char*pr=c>3?v[3]:"/tmp/rdo.ptx";DIM=c>4?atoi(v[4]):8;BITS=(DIM==16)?4:3;
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext cx;CK(cuDevicePrimaryCtxRetain(&cx,d));CK(cuCtxSetCurrent(cx));
 CUmodule m64,m16,mr;CK(cuModuleLoad(&m64,p64));CK(cuModuleLoad(&m16,p16));CK(cuModuleLoad(&mr,pr));
 CK(cuModuleGetFunction(&F64,m64,"step"));CK(cuModuleGetFunction(&F16,m16,"step"));CK(cuModuleGetFunction(&RDO,mr,"step"));
 CK(cuMemAlloc(&dA_,DIM*8));CK(cuMemAlloc(&dB_,DIM*8));CK(cuMemAlloc(&dx_,16*8));CK(cuMemAlloc(&dH_,DIM*16*8));
 CK(cuMemAlloc(&dSf,256*8));CK(cuMemAlloc(&dS4,256*4));CK(cuMemAlloc(&dC_,DIM*8));CK(cuMemAlloc(&dHd_,256*8));CK(cuMemAlloc(&dy_,16*8));
 double A[16]={0},B[16]={0},x[16],Hsm[256],C[16]={0};
 for(int i=0;i<DIM;i++){A[i]=0.3*((i%2)?-1:1)+0.04*i;B[i]=0.12-0.02*i;C[i]=0.25*((i%2)?1:-1)+0.03*i;}
 for(int b=0;b<16;b++)x[b]=0.3*sin(0.5*b);
 for(int b=0;b<16;b++)for(int j=0;j<DIM;j++)Hsm[b*DIM+j]=0.2*sin(0.4*j+0.6*b);

 // ── Part 1: forward exactness ────────────────────────────────────────────────────────────────
 double Sref[256];ref_step(A,B,x,Hsm,Sref);
 CK(cuMemcpyHtoD(dA_,A,DIM*8));CK(cuMemcpyHtoD(dB_,B,DIM*8));CK(cuMemcpyHtoD(dx_,x,16*8));CK(cuMemcpyHtoD(dH_,Hsm,DIM*16*8));
 stepf64(dA_,dB_,dx_,dH_,dSf);double Sf[256];CK(cuMemcpyDtoH(Sf,dSf,256*8));
 stepf16(dA_,dB_,dx_,dH_,dS4);float S4[256];CK(cuMemcpyDtoH(S4,dS4,256*4));
 double mf64=0,mf16=0;for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){
   double e64=fabs(Sf[k*16+b]-Sref[k*16+b]);if(e64>mf64)mf64=e64;
   double e16=fabs((double)S4[k*16+b]-Sref[k*16+b]);if(e16>mf16)mf16=e16;}
 printf("Part 1 — forward S = A⊗H + B·x (%s), maxerr vs host f64:\n",(DIM==16)?"sedenion":"octonion");
 printf("  f64 step (ossm_oct_step_f64): %.3e   ← EXACT\n",mf64);
 printf("  f16 tile (ossm_oct_step):     %.3e   ← lossy tensor-core tile\n",mf16);
 if(mf64>1e-12){printf("FAIL: f64 step not exact\n");return 1;}
 printf("  PASS (f64 forward is exact to machine precision)\n");

 // ── Part 2: T=1 full-cell FD gradient check (f64 vs f16 forward) ─────────────────────────────
 CK(cuMemcpyHtoD(dC_,C,DIM*8));
 double tgt[16],At[16];for(int i=0;i<DIM;i++)At[i]=A[i]+0.05;
 cell_loss(At,B,x,Hsm,C,0,1,tgt);   // teacher y at a shifted A (via the f64 forward)
 // analytic dA (host f64, T=1)
 double S[256];ref_step(A,B,x,Hsm,S);
 double h[256],y[16];for(int p=0;p<DIM;p++)for(int b=0;b<16;b++)h[p*16+b]=sig(S[p*16+b]);
 for(int b=0;b<16;b++){double s=0;for(int p=0;p<DIM;p++)s+=(double)cds(p,p,BITS)*C[p]*h[p*16+b];y[b]=s;}
 double dS[256];for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){
   double dy=2.0*(y[b]-tgt[b]); double dh=dy*(double)cds(k,k,BITS)*C[k];
   dS[k*16+b]=dh*sigp(S[k*16+b]);}
 double dAan[16];for(int q=0;q<DIM;q++){double s=0;
   for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){int m=k^q;s+=(double)cds(q,m,BITS)*Hsm[b*DIM+m]*dS[k*16+b];}
   dAan[q]=s;}
 // central FD via each forward
 double eps=1e-3, fd64_err=0, fd16_err=0;
 for(int q=0;q<DIM;q++){
   double Ap[16];memcpy(Ap,A,128);
   Ap[q]=A[q]+eps;double Lp64=cell_loss(Ap,B,x,Hsm,C,tgt,1,0),Lp16=cell_loss(Ap,B,x,Hsm,C,tgt,0,0);
   Ap[q]=A[q]-eps;double Lm64=cell_loss(Ap,B,x,Hsm,C,tgt,1,0),Lm16=cell_loss(Ap,B,x,Hsm,C,tgt,0,0);
   double fd64=(Lp64-Lm64)/(2*eps),fd16=(Lp16-Lm16)/(2*eps);
   double e64=fabs(fd64-dAan[q]);if(e64>fd64_err)fd64_err=e64;
   double e16=fabs(fd16-dAan[q]);if(e16>fd16_err)fd16_err=e16;}
 printf("Part 2 — T=1 full-cell FD gradient check (dA), maxerr vs analytic f64 BPTT:\n");
 printf("  FD via f64 forward: %.3e   ← clean/sharp\n",fd64_err);
 printf("  FD via f16 forward: %.3e   ← noisy (f16 quantization)\n",fd16_err);
 int ok = (fd64_err < 1e-3) && (fd64_err < fd16_err);
 if(ok){printf("PASS: the exact f64 forward yields a clean finite-difference gradient (f64 << f16 error) on GB10\n");return 0;}
 printf("FAIL: f64 FD not clean / not better than f16\n");return 1;
}

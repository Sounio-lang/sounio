// BPTT on F64 TENSOR CORES: trains the linear recurrence H_t = A⊗H_{t-1} + B·x_t on the DGX Spark
// GB10 with BOTH matrix-multiplies running on f64 tensor cores (wmma m8n8k4.f64) — the forward
// L(A)·H via oct_batch_mul_f64 and the backward recurrent gradient L(A)ᵀ·dS via oct_batch_mul_f64_t.
// The tensor-core products are exact to machine precision, so the gradients are exact and Adam drives
// the loss to ~0. B·x, the parameter gradients (da, dB), the dY and the layout transposes are host
// arithmetic (not tensor-core ops). Usage: run_bptt_f64tc <mul.ptx> <mulT.ptx> <dim>.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
static int DIM,BITS,ND;
static CUfunction MUL,MULT;
static CUdeviceptr dA_,dB_,dSm_,dout_;
// D = L(A)·B (forward tile) or L(A)ᵀ·B (transpose tile), B state-major → D D-layout f64.
static void tile(CUfunction f,CUdeviceptr A,CUdeviceptr Bsm,CUdeviceptr D){void*a[]={&A,&Bsm,&D};cuLaunchKernel(f,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}

int main(int c,char**v){
 const char*pm=c>1?v[1]:"/tmp/mul.ptx";const char*pt=c>2?v[2]:"/tmp/mulT.ptx";DIM=c>3?atoi(v[3]):8;BITS=(DIM==16)?4:3;ND=DIM*16;
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext cx;CK(cuDevicePrimaryCtxRetain(&cx,d));CK(cuCtxSetCurrent(cx));
 CUmodule mm,mt;CK(cuModuleLoad(&mm,pm));CK(cuModuleLoad(&mt,pt));
 CK(cuModuleGetFunction(&MUL,mm,"step"));CK(cuModuleGetFunction(&MULT,mt,"step"));
 CK(cuMemAlloc(&dA_,DIM*8));CK(cuMemAlloc(&dB_,16*DIM*8));CK(cuMemAlloc(&dSm_,16*DIM*8));CK(cuMemAlloc(&dout_,256*8));

 // ── Part 1: transposed tile L(A)ᵀ·B exactness vs host f64 ────────────────────────────────────
 {
  double A[16]={0},Bsm[256]={0};
  for(int i=0;i<DIM;i++)A[i]=0.4*((i%2)?-1:1)+0.05*i;
  for(int st=0;st<16;st++)for(int j=0;j<DIM;j++)Bsm[st*DIM+j]=0.2*sin(0.4*j+0.6*st);
  double ref[256]={0};
  for(int comp=0;comp<DIM;comp++)for(int st=0;st<16;st++){double s=0;
    for(int j=0;j<DIM;j++)s+=(double)cds(j^comp,comp,BITS)*A[j^comp]*Bsm[st*DIM+j];ref[comp*16+st]=s;}
  CK(cuMemcpyHtoD(dA_,A,DIM*8));CK(cuMemcpyHtoD(dSm_,Bsm,ND*8));double z[256]={0};CK(cuMemcpyHtoD(dout_,z,256*8));
  tile(MULT,dA_,dSm_,dout_);double D[256];CK(cuMemcpyDtoH(D,dout_,256*8));
  double mx=0;int f=0;for(int comp=0;comp<DIM;comp++)for(int st=0;st<16;st++){double e=fabs(D[comp*16+st]-ref[comp*16+st]);if(e>mx)mx=e;if(e>1e-12)f++;}
  printf("Part 1 — transposed f64 tile L(A)ᵀ·B (%s): mismatch %d/%d maxerr=%.3e\n",(DIM==16)?"sedenion":"octonion",f,DIM*16,mx);
  if(f){printf("FAIL (part 1)\n");return 1;}printf("  PASS (exact)\n");
 }

 // ── Part 2: BPTT training (both matmuls on f64 tensor cores) + Adam ──────────────────────────
 const int T=3,NIT=120; const double lr=0.02,b1=0.9,b2=0.999,eps=1e-8;
 double h0[256],x[3][16],As[16],Bs[16];
 for(int st=0;st<16;st++)for(int j=0;j<DIM;j++)h0[st*DIM+j]=0.1*((st%2)?-1:1)+0.02*j;
 for(int t=0;t<T;t++)for(int st=0;st<16;st++)x[t][st]=0.3*sin(0.7*t+0.4*st);
 double asc=(DIM==16)?0.5:1.0;
 for(int i=0;i<DIM;i++){As[i]=asc*(0.22*((i%2)?-1:1)+0.03*i);Bs[i]=asc*(0.12-0.02*i);}
 // fwd helper (host-side orchestration of the f64 tile + B·x)
 // teacher targets tgt[t] (D-layout) from the teacher trajectory
 double tgt[3][256];
 { double hsm[256];memcpy(hsm,h0,sizeof(double)*256);
   CK(cuMemcpyHtoD(dA_,As,DIM*8));
   for(int t=0;t<T;t++){ CK(cuMemcpyHtoD(dSm_,hsm,ND*8)); tile(MUL,dA_,dSm_,dout_);
     double D[256];CK(cuMemcpyDtoH(D,dout_,256*8));
     for(int k=0;k<DIM;k++)for(int st=0;st<16;st++)tgt[t][k*16+st]=D[k*16+st]+Bs[k]*x[t][st];
     double nh[256]={0};for(int k=0;k<DIM;k++)for(int st=0;st<16;st++)nh[st*DIM+k]=tgt[t][k*16+st];memcpy(hsm,nh,sizeof(double)*256);}}
 double A[16],Bp[16];for(int i=0;i<DIM;i++){A[i]=As[i]+0.08*((i%2)?-1:1);Bp[i]=Bs[i]+0.07*((i%3)?1:-1);}
 double mA[16]={0},vA[16]={0},mB[16]={0},vB[16]={0};
 double loss0=0,lossN=0;
 for(int it=1; it<=NIT; it++){
  double hsm[4][256],Straj[3][256];memcpy(hsm[0],h0,sizeof(double)*256);
  CK(cuMemcpyHtoD(dA_,A,DIM*8));
  double loss=0;
  for(int t=0;t<T;t++){ CK(cuMemcpyHtoD(dSm_,hsm[t],ND*8)); tile(MUL,dA_,dSm_,dout_);
    double D[256];CK(cuMemcpyDtoH(D,dout_,256*8));
    for(int k=0;k<DIM;k++)for(int st=0;st<16;st++){Straj[t][k*16+st]=D[k*16+st]+Bp[k]*x[t][st];
      double e=Straj[t][k*16+st]-tgt[t][k*16+st];loss+=e*e;}
    for(int k=0;k<DIM;k++)for(int st=0;st<16;st++)hsm[t+1][st*DIM+k]=Straj[t][k*16+st];}
  if(it==1)loss0=loss; lossN=loss;
  // backward: dHnext(D-layout)=0; dA,dB accumulate
  double dHnext[256]={0},dAacc[16]={0},dBacc[16]={0};
  for(int t=T-1;t>=0;t--){
   double dHt[256];for(int k=0;k<DIM;k++)for(int st=0;st<16;st++)dHt[k*16+st]=2.0*(Straj[t][k*16+st]-tgt[t][k*16+st])+dHnext[k*16+st];
   // dB += Σ dHt·x
   for(int k=0;k<DIM;k++){double s=0;for(int st=0;st<16;st++)s+=dHt[k*16+st]*x[t][st];dBacc[k]+=s;}
   // dA += da(H_{t}, dHt): H_{t}=hsm[t] state-major; dA[q]=Σ σ(q,k⊕q) hsm[t][b*dim+(k⊕q)] dHt[k,b]
   for(int q=0;q<DIM;q++){double s=0;for(int k=0;k<DIM;k++)for(int st=0;st<16;st++){int m=k^q;s+=(double)cds(q,m,BITS)*hsm[t][st*DIM+m]*dHt[k*16+st];}dAacc[q]+=s;}
   // dHprev = L(A)ᵀ·dHt on the f64 tensor-core transpose tile (dHt provided state-major)
   double dHt_sm[256]={0};for(int k=0;k<DIM;k++)for(int st=0;st<16;st++)dHt_sm[st*DIM+k]=dHt[k*16+st];
   CK(cuMemcpyHtoD(dSm_,dHt_sm,ND*8)); tile(MULT,dA_,dSm_,dout_); CK(cuMemcpyDtoH(dHnext,dout_,256*8));
  }
  // Adam
  double bc1=1.0-pow(b1,it),bc2=1.0-pow(b2,it);
  for(int p=0;p<DIM;p++){
   mA[p]=b1*mA[p]+(1-b1)*dAacc[p];vA[p]=b2*vA[p]+(1-b2)*dAacc[p]*dAacc[p];A[p]-=lr*(mA[p]/bc1)/(sqrt(vA[p]/bc2)+eps);
   mB[p]=b1*mB[p]+(1-b1)*dBacc[p];vB[p]=b2*vB[p]+(1-b2)*dBacc[p]*dBacc[p];Bp[p]-=lr*(mB[p]/bc1)/(sqrt(vB[p]/bc2)+eps);}
  if(it==1||it==NIT||it%30==0)printf("  iter %3d  loss %.6e\n",it,loss);
 }
 printf("Part 2 — BPTT on f64 tensor cores (fwd L(A)·H + bwd L(A)ᵀ·dS, both m8n8k4.f64) + Adam:\n");
 printf("  loss %.6e → %.6e  (%.1f%% reduction)\n",loss0,lossN,100.0*(loss0-lossN)/loss0);
 if(lossN<1e-4*loss0){printf("PASS: trained the recurrence with both matmuls on exact f64 tensor cores → loss ~0 on GB10\n");return 0;}
 printf("FAIL: loss did not fall enough\n");return 1;
}

// END-TO-END training THROUGH the associator on the DGX Spark GB10. Model: y = w·[a,b,c], with
// LEARNABLE octonion weights a, b and a linear readout w. The associator [a,b,c] and its reverse-mode
// gradients (da, db) run on the compiler-lowered tensor-core kernels (oct_assoc forward; ossm_oct_bwd +
// oct_batch_mul for the backward). Teacher-generated targets (reachable); Adam. Shows the associator is
// a trainable layer — the loss falls through the associator's VJP.
// Usage: run_assoc_train <oct_assoc.ptx> <ossm_oct_bwd.ptx> <oct_batch_mul.ptx>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda.h>
static unsigned RS=777; static double rnd(){RS=RS*1664525u+1013904223u; return ((RS>>8)&0xffffff)/16777216.0*2-1;}
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
static const int BITS=3;
static CUfunction ASS,BWD,MUL;
static void assoc(CUdeviceptr a,CUdeviceptr b,CUdeviceptr H,CUdeviceptr o){void*A[]={&a,&b,&H,&o};cuLaunchKernel(ASS,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();}
static void bwd(CUdeviceptr a,CUdeviceptr Hp,CUdeviceptr dHt,CUdeviceptr dHp,CUdeviceptr dA){void*A[]={&a,&Hp,&dHt,&dHp,&dA};cuLaunchKernel(BWD,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();}
static void mul(CUdeviceptr a,CUdeviceptr H,CUdeviceptr o){void*A[]={&a,&H,&o};cuLaunchKernel(MUL,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();}
static CUdeviceptr da_,db_,dH_,dO_,dHt_,dHpr_,dAacc_,dq_,dMul_;
// z = oct_assoc(a,b,C) → D-layout f32 [k*16+i]
static void fwd_assoc(const double*a,const double*b,const double*C,float*z){
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(db_,b,64);cuMemcpyHtoD(dH_,C,128*8);float zz[256]={0};cuMemcpyHtoD(dO_,zz,256*4);
 assoc(da_,db_,dH_,dO_);cuMemcpyDtoH(z,dO_,256*4);
}
// accumulate da_g, db_g for one group into gda,gdb given upstream dD (D-layout f64)
static void grad_ab(const double*a,const double*b,const double*C,const double*dD,double*gda,double*gdb){
 double z8[8]={0};
 // dP + dq: ossm_oct_bwd(a, C, dD)
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(dH_,C,128*8);cuMemcpyHtoD(dHt_,dD,256*8);cuMemcpyHtoD(dAacc_,z8,64);
 bwd(da_,dH_,dHt_,dHpr_,dAacc_);
 double dP[8];cuMemcpyDtoH(dP,dAacc_,64); double dhp[256];cuMemcpyDtoH(dhp,dHpr_,256*8);
 double dq[256];for(int i=0;i<256;i++)dq[i]=-dhp[i];
 // q = oct_batch_mul(b,C) → transpose to state-major
 cuMemcpyHtoD(da_,b,64);cuMemcpyHtoD(dH_,C,128*8);float zz[256]={0};cuMemcpyHtoD(dMul_,zz,256*4);
 mul(da_,dH_,dMul_);float qf[256];cuMemcpyDtoH(qf,dMul_,256*4);
 double qsm[128];for(int k=0;k<8;k++)for(int i=0;i<16;i++)qsm[i*8+k]=qf[k*16+i];
 // da_from_r = −ossm_oct_bwd(a, qsm, dD).dA
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(dH_,qsm,128*8);cuMemcpyHtoD(dHt_,dD,256*8);cuMemcpyHtoD(dAacc_,z8,64);
 bwd(da_,dH_,dHt_,dHpr_,dAacc_);double dar[8];cuMemcpyDtoH(dar,dAacc_,64);
 // db_from_q = ossm_oct_bwd(a, C, dq).dA
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(dH_,C,128*8);cuMemcpyHtoD(dHt_,dq,256*8);cuMemcpyHtoD(dAacc_,z8,64);
 bwd(da_,dH_,dHt_,dHpr_,dAacc_);double dbq[8];cuMemcpyDtoH(dbq,dAacc_,64);
 for(int i=0;i<8;i++){double s=0;for(int j=0;j<8;j++)s+=(double)cds(i,j,BITS)*b[j]*dP[i^j];gda[i]+=s-dar[i];}
 for(int j=0;j<8;j++){double s=0;for(int i=0;i<8;i++)s+=(double)cds(i,j,BITS)*a[i]*dP[i^j];gdb[j]+=s+dbq[j];}
}
int main(int c,char**v){
 const char*pa=c>1?v[1]:"/tmp/assoc.ptx";const char*pbw=c>2?v[2]:"/tmp/bwd.ptx";const char*pm=c>3?v[3]:"/tmp/mul.ptx";
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule ma,mb,mm;CK(cuModuleLoad(&ma,pa));CK(cuModuleLoad(&mb,pbw));CK(cuModuleLoad(&mm,pm));
 CK(cuModuleGetFunction(&ASS,ma,"step"));CK(cuModuleGetFunction(&BWD,mb,"step"));CK(cuModuleGetFunction(&MUL,mm,"step"));
 CK(cuMemAlloc(&da_,64));CK(cuMemAlloc(&db_,64));CK(cuMemAlloc(&dH_,128*8));CK(cuMemAlloc(&dO_,256*4));
 CK(cuMemAlloc(&dHt_,256*8));CK(cuMemAlloc(&dHpr_,256*8));CK(cuMemAlloc(&dAacc_,64));CK(cuMemAlloc(&dq_,128*8));CK(cuMemAlloc(&dMul_,256*4));
 const int G=8; double C[8][128];
 for(int g=0;g<G;g++)for(int i=0;i<16;i++)for(int k=0;k<8;k++)C[g][i*8+k]=rnd();
 double as[8],bs[8],ws[8]; for(int k=0;k<8;k++){as[k]=0.4*rnd();bs[k]=0.4*rnd();ws[k]=rnd();}
 // teacher targets
 double tgt[8][16];
 for(int g=0;g<G;g++){float z[256];fwd_assoc(as,bs,C[g],z);
   for(int i=0;i<16;i++){double y=0;for(int k=0;k<8;k++)y+=ws[k]*(double)z[k*16+i];tgt[g][i]=y;}}
 // student init = teacher + perturbation
 double a[8],b[8],w[8];for(int k=0;k<8;k++){a[k]=as[k]+0.15*rnd();b[k]=bs[k]+0.15*rnd();w[k]=ws[k]+0.15*rnd();}
 double mA[8]={0},vA[8]={0},mB[8]={0},vB[8]={0},mW[8]={0},vW[8]={0};
 const double lr=0.02,b1=0.9,b2=0.999,eps=1e-8; const int NIT=300;
 double loss0=0,lossN=0;
 for(int it=1;it<=NIT;it++){
  double gda[8]={0},gdb[8]={0},gdw[8]={0},loss=0;
  for(int g=0;g<G;g++){
   float z[256];fwd_assoc(a,b,C[g],z);
   double dD[256]={0};
   for(int i=0;i<16;i++){double y=0;for(int k=0;k<8;k++)y+=w[k]*(double)z[k*16+i];
     double dy=2.0*(y-tgt[g][i]);loss+=(y-tgt[g][i])*(y-tgt[g][i]);
     for(int k=0;k<8;k++){gdw[k]+=dy*(double)z[k*16+i]; dD[k*16+i]=dy*w[k];}}
   grad_ab(a,b,C[g],dD,gda,gdb);
  }
  if(it==1)loss0=loss; lossN=loss;
  double bc1=1.0-pow(b1,it),bc2=1.0-pow(b2,it);
#define ADAM(TH,G,M,V) for(int k=0;k<8;k++){M[k]=b1*M[k]+(1-b1)*G[k];V[k]=b2*V[k]+(1-b2)*G[k]*G[k];TH[k]-=lr*(M[k]/bc1)/(sqrt(V[k]/bc2)+eps);}
  ADAM(a,gda,mA,vA) ADAM(b,gdb,mB,vB) ADAM(w,gdw,mW,vW)
#undef ADAM
  if(it==1||it==NIT||it%50==0)printf("  iter %3d  loss %.6e\n",it,loss);
 }
 printf("End-to-end training THROUGH the associator (learn a,b,w; associator forward + da/db on tensor cores) + Adam:\n");
 printf("  loss %.6e → %.6e  (%.2f%% reduction, %d samples)\n",loss0,lossN,100.0*(loss0-lossN)/loss0,G*16);
 if(lossN<1e-3*loss0){printf("PASS: the associator is a TRAINABLE tensor-core layer — the loss falls through its VJP on GB10\n");return 0;}
 printf("FAIL\n");return 1;
}

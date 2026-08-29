// Associator VJP w.r.t. the WEIGHTS a, b — completes the associator gradient. Given upstream dD of
// [a,b,H] = (a⊗b)⊗H − a⊗(b⊗H), compute da, db. They DECOMPOSE into the already-merged tensor-core
// kernels + host glue:
//   dP = da_accum(H,dD)          = ossm_oct_bwd(a,H,dD).dA
//   dq = −L(a)ᵀ·dD               = −ossm_oct_bwd(a,H,dD).dHprev
//   q  = L(b)·H                  = oct_batch_mul(b,H)
//   da_from_r = −da_accum(q,dD)  = −ossm_oct_bwd(·,qᵀ,dD).dA
//   db_from_q =  da_accum(H,dq)  =  ossm_oct_bwd(·,H,dq).dA
//   da_from_P[i]=Σ_j σ(i,j)b[j]dP[i⊕j] ; db_from_P[j]=Σ_i σ(i,j)a[i]dP[i⊕j]   (a⊗b product VJP, host)
//   da = da_from_P + da_from_r ; db = db_from_P + db_from_q
// Validated vs the analytic host da/db. Usage: run_assoc_dadb <ossm_oct_bwd.ptx> <oct_batch_mul.ptx>
#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
static const int DIM=8, BITS=3;
static CUfunction BWD,MUL;
// ossm_oct_bwd(pA,pHprev[state-major],pdHt[D-layout],pdHprev[D-layout out],pdA[accumulate])
static void bwd(CUdeviceptr A,CUdeviceptr Hp,CUdeviceptr dHt,CUdeviceptr dHp,CUdeviceptr dA){void*a[]={&A,&Hp,&dHt,&dHp,&dA};cuLaunchKernel(BWD,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
// oct_batch_mul(pa,pH[state-major],pout[D-layout f32])
static void mul(CUdeviceptr a_,CUdeviceptr H,CUdeviceptr o){void*a[]={&a_,&H,&o};cuLaunchKernel(MUL,1,1,1,32,1,1,0,0,a,0);cuCtxSynchronize();}
int main(int c,char**v){
 const char*pb=c>1?v[1]:"/tmp/bwd.ptx"; const char*pm=c>2?v[2]:"/tmp/mul.ptx";
 double a[8],b[8],H[128],dD[256];
 for(int i=0;i<8;i++){a[i]=0.35*((i%2)?-1:1)+0.05*i; b[i]=0.25-0.04*i+((i%3)?0.1:-0.1);}
 for(int bb=0;bb<16;bb++)for(int j=0;j<8;j++)H[bb*8+j]=0.2*sin(0.4*j+0.6*bb);     // state-major
 for(int k=0;k<8;k++)for(int bb=0;bb<16;bb++)dD[k*16+bb]=0.2*sin(0.5*k+0.3*bb);   // D-layout
 // ── analytic host da, db ──
 double P[8]={0}; for(int i=0;i<8;i++)for(int j=0;j<8;j++)P[i^j]+=(double)cds(i,j,BITS)*a[i]*b[j];
 double dP[8]={0};
 for(int p=0;p<8;p++){double s=0;for(int k=0;k<8;k++)for(int bb=0;bb<16;bb++){int m=k^p;s+=(double)cds(p,m,BITS)*H[bb*8+m]*dD[k*16+bb];}dP[p]=s;}
 double q[128],dq[256];
 for(int k=0;k<8;k++)for(int bb=0;bb<16;bb++){double s=0;for(int l=0;l<8;l++)s+=(double)cds(k^l,l,BITS)*b[k^l]*H[bb*8+l];q[bb*8+k]=s;} // q state-major
 for(int k=0;k<8;k++)for(int bb=0;bb<16;bb++){double s=0;for(int mm=0;mm<8;mm++)s+=(double)cds(mm^k,k,BITS)*a[mm^k]*dD[mm*16+bb];dq[k*16+bb]=-s;} // dq=−L(a)ᵀdD, D-layout
 double da_r[8]={0},db_q[8]={0};
 for(int p=0;p<8;p++){double s=0;for(int k=0;k<8;k++)for(int bb=0;bb<16;bb++){int m=k^p;s+=(double)cds(p,m,BITS)*q[bb*8+m]*dD[k*16+bb];}da_r[p]=-s;}
 for(int p=0;p<8;p++){double s=0;for(int k=0;k<8;k++)for(int bb=0;bb<16;bb++){int m=k^p;s+=(double)cds(p,m,BITS)*H[bb*8+m]*dq[k*16+bb];}db_q[p]=s;}
 double da_ref[8],db_ref[8];
 for(int i=0;i<8;i++){double s=0;for(int j=0;j<8;j++)s+=(double)cds(i,j,BITS)*b[j]*dP[i^j];da_ref[i]=s+da_r[i];}
 for(int j=0;j<8;j++){double s=0;for(int i=0;i<8;i++)s+=(double)cds(i,j,BITS)*a[i]*dP[i^j];db_ref[j]=s+db_q[j];}
 // ── orchestrate merged kernels ──
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule mb,mm;CK(cuModuleLoad(&mb,pb));CK(cuModuleLoad(&mm,pm));
 CK(cuModuleGetFunction(&BWD,mb,"step"));CK(cuModuleGetFunction(&MUL,mm,"step"));
 CUdeviceptr dA_,dHp_,ddHt_,ddHpr_,ddA_,dq_,dMul_;
 CK(cuMemAlloc(&dA_,8*8));CK(cuMemAlloc(&dHp_,128*8));CK(cuMemAlloc(&ddHt_,256*8));CK(cuMemAlloc(&ddHpr_,256*8));CK(cuMemAlloc(&ddA_,8*8));CK(cuMemAlloc(&dq_,128*8));CK(cuMemAlloc(&dMul_,256*4));
 double z8[8]={0}; float z4[256]={0};
 // Call 1: ossm_oct_bwd(a, H, dD) → dHprev1=L(a)ᵀ·dD, dA1=dP
 CK(cuMemcpyHtoD(dA_,a,64));CK(cuMemcpyHtoD(dHp_,H,128*8));CK(cuMemcpyHtoD(ddHt_,dD,256*8));CK(cuMemcpyHtoD(ddA_,z8,64));
 bwd(dA_,dHp_,ddHt_,ddHpr_,ddA_);
 double kdP[8]; CK(cuMemcpyDtoH(kdP,ddA_,64));
 double khp[256]; CK(cuMemcpyDtoH(khp,ddHpr_,256*8)); double kdq[256]; for(int i=0;i<256;i++)kdq[i]=-khp[i];
 // q = oct_batch_mul(b, H) → D-layout f32; transpose to state-major
 CK(cuMemcpyHtoD(dA_,b,64));CK(cuMemcpyHtoD(dHp_,H,128*8));CK(cuMemcpyHtoD(dMul_,z4,256*4));
 mul(dA_,dHp_,dMul_); float qf[256]; CK(cuMemcpyDtoH(qf,dMul_,256*4));
 double qsm[128]; for(int k=0;k<8;k++)for(int bb=0;bb<16;bb++)qsm[bb*8+k]=qf[k*16+bb];
 // Call 2: ossm_oct_bwd(a, qsm, dD) → dA2 = da_accum(q,dD); da_from_r = −dA2
 CK(cuMemcpyHtoD(dA_,a,64));CK(cuMemcpyHtoD(dHp_,qsm,128*8));CK(cuMemcpyHtoD(ddHt_,dD,256*8));CK(cuMemcpyHtoD(ddA_,z8,64));
 bwd(dA_,dHp_,ddHt_,ddHpr_,ddA_); double kdar[8]; CK(cuMemcpyDtoH(kdar,ddA_,64)); for(int i=0;i<8;i++)kdar[i]=-kdar[i];
 // Call 3: ossm_oct_bwd(a, H, dq) → dA3 = da_accum(H,dq) = db_from_q
 CK(cuMemcpyHtoD(dA_,a,64));CK(cuMemcpyHtoD(dHp_,H,128*8));CK(cuMemcpyHtoD(ddHt_,kdq,256*8));CK(cuMemcpyHtoD(ddA_,z8,64));
 bwd(dA_,dHp_,ddHt_,ddHpr_,ddA_); double kdbq[8]; CK(cuMemcpyDtoH(kdbq,ddA_,64));
 // host product-VJP over the (exact-f64) dP from the kernel + combine
 double da[8],db[8];
 for(int i=0;i<8;i++){double s=0;for(int j=0;j<8;j++)s+=(double)cds(i,j,BITS)*b[j]*kdP[i^j];da[i]=s+kdar[i];}
 for(int j=0;j<8;j++){double s=0;for(int i=0;i<8;i++)s+=(double)cds(i,j,BITS)*a[i]*kdP[i^j];db[j]=s+kdbq[j];}
 double mda=0,mdb=0,mdp=0; for(int i=0;i<8;i++){mdp=fmax(mdp,fabs(kdP[i]-dP[i]));mda=fmax(mda,fabs(da[i]-da_ref[i]));mdb=fmax(mdb,fabs(db[i]-db_ref[i]));}
 printf("Associator weight-gradient VJP (da, db) via merged kernels + host glue, vs analytic (GB10):\n");
 printf("  dP = da_accum(H,dD)  (f64, exact):   maxerr = %.2e\n",mdp);
 printf("  da = productVJP_a(b,dP) − da_accum(q,dD):   maxerr = %.4f\n",mda);
 printf("  db = productVJP_b(a,dP) + da_accum(H,dq):   maxerr = %.4f\n",mdb);
 if(mdp<1e-6 && mda<0.05 && mdb<0.05){printf("PASS: the associator's weight gradients da, db decompose into the merged tensor-core kernels and match the analytic VJP on GB10\n");return 0;}
 printf("FAIL\n");return 1;
}

// Validates the associator VJP-w.r.t.-H kernel dH = L(a⊗b)ᵀ·dD − L(b)ᵀ·(L(a)ᵀ·dD) on the DGX Spark
// GB10, vs the analytic host reference. Output dH is f32 D-layout [comp*16+batch]; tolerance covers the
// composed f16 tiles. Usage: run_assoc_bwd_dh <ptx> <dim>.
#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c,char**v){
 const char*p=c>1?v[1]:"/tmp/abd.ptx"; int DIM=c>2?atoi(v[2]):8; int BITS=(DIM==16)?4:3;
 double a[16]={0},b[16]={0},dD[256]={0};
 for(int i=0;i<DIM;i++){a[i]=0.35*((i%2)?-1:1)+0.05*i; b[i]=0.25-0.04*i+((i%3)?0.1:-0.1);}
 for(int k=0;k<DIM;k++)for(int bb=0;bb<16;bb++)dD[k*16+bb]=0.2*sin(0.5*k+0.3*bb);
 // host analytic: P=a⊗b; LP[k][j]=σ(k⊕j,j)P[k⊕j]; La[k][m]=σ(k⊕m,m)a[k⊕m]; Lb[m][j]=σ(m⊕j,j)b[m⊕j];
 //   dH[j][bb] = Σ_k (LP[k][j] − Σ_m La[k][m]Lb[m][j]) · dD[k*16+bb]
 double P[16]={0}; for(int i=0;i<DIM;i++)for(int j=0;j<DIM;j++)P[i^j]+=(double)cds(i,j,BITS)*a[i]*b[j];
 double ref[256]={0}, mref=0;
 for(int j=0;j<DIM;j++)for(int bb=0;bb<16;bb++){double s=0;
   for(int k=0;k<DIM;k++){
     double lp=(double)cds(k^j,j,BITS)*P[k^j];
     double lab=0; for(int m=0;m<DIM;m++) lab+=(double)cds(k^m,m,BITS)*a[k^m]*(double)cds(m^j,j,BITS)*b[m^j];
     s+=(lp-lab)*dD[k*16+bb];
   }
   ref[j*16+bb]=s; if(fabs(s)>mref)mref=fabs(s);}
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr da,db,dd,dH;CK(cuMemAlloc(&da,DIM*8));CK(cuMemAlloc(&db,DIM*8));CK(cuMemAlloc(&dd,256*8));CK(cuMemAlloc(&dH,256*4));
 CK(cuMemcpyHtoD(da,a,DIM*8));CK(cuMemcpyHtoD(db,b,DIM*8));CK(cuMemcpyHtoD(dd,dD,256*8));float z[256]={0};CK(cuMemcpyHtoD(dH,z,256*4));
 void*args[]={&da,&db,&dd,&dH};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 float H[256];CK(cuMemcpyDtoH(H,dH,256*4));
 double mx=0;int fails=0;double tol=0.05*mref+0.02;
 for(int j=0;j<DIM;j++)for(int bb=0;bb<16;bb++){double e=fabs((double)H[j*16+bb]-ref[j*16+bb]);if(e>mx)mx=e;if(e>tol)fails++;}
 const char*nm=(DIM==16)?"SEDENION":"octonion";
 printf("Associator VJP dH = L(a⊗b)ᵀ·dD − L(b)ᵀ(L(a)ᵀ·dD) (%s) on GB10: mismatch %d/%d maxerr=%.4f (|dH|max=%.3f, tol=%.3f)\n",nm,fails,DIM*16,mx,mref,tol);
 if(!fails){printf("PASS: the associator is differentiable w.r.t. its input — dH matches the analytic Jacobian transpose on GB10\n");return 0;}
 printf("FAIL\n");return 1;
}

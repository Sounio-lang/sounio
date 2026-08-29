// Driver-API harness for the compiler-lowered VJP/backward kernel `step` of D = L(a)·H:
//   dH[b*dim+j] = Σ_k σ(k⊕j,j)·a[k⊕j]·dD[k,b]        (tensor-core tile, L(a)ᵀ·dD)
//   da[p]       = Σ_{k,b} σ(p,k⊕p)·H[k⊕p,b]·dD[k,b]  (runtime f64 accumulation)
// Usage: run_vjp <ptx> <dim>  (dim=8 octonion / 16 sedenion). Validates vs the analytic Jacobian
// transpose on the DGX Spark GB10. dH goes through the f16 tile (looser tol); da is full f64.
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
 const char*p=c>1?v[1]:"/tmp/vjp.ptx"; int dim=c>2?atoi(v[2]):8; int bits=(dim==16)?4:3;
 double a[16]={0}, H[256]={0}, dD[256]={0};
 for(int i=0;i<dim;i++)a[i]=0.4*((i%2)?-1:1)+0.05*i;
 for(int b=0;b<16;b++)for(int j=0;j<dim;j++)H[b*dim+j]=((b+1)*0.11+j*0.05)*((j%2)?-1:1);
 for(int k=0;k<dim;k++)for(int b=0;b<16;b++)dD[k*16+b]=0.2*((k+b)%3-1)+0.03*k-0.02*b;
 // analytic VJP
 double dH_ref[256]={0}, da_ref[16]={0};
 for(int b=0;b<16;b++)for(int j=0;j<dim;j++){double s=0;
   for(int k=0;k<dim;k++)s+=(double)cds(k^j,j,bits)*a[k^j]*dD[k*16+b];
   dH_ref[b*dim+j]=s;}
 for(int pp=0;pp<dim;pp++){double s=0;
   for(int k=0;k<dim;k++)for(int b=0;b<16;b++){int m=k^pp; s+=(double)cds(pp,m,bits)*H[b*dim+m]*dD[k*16+b];}
   da_ref[pp]=s;}
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr pa,pH,pdD,pdH,pda;
 CK(cuMemAlloc(&pa,dim*8));CK(cuMemAlloc(&pH,16*dim*8));CK(cuMemAlloc(&pdD,256*8));
 CK(cuMemAlloc(&pdH,256*8));CK(cuMemAlloc(&pda,16*8));
 CK(cuMemcpyHtoD(pa,a,dim*8));CK(cuMemcpyHtoD(pH,H,16*dim*8));CK(cuMemcpyHtoD(pdD,dD,256*8));
 double z[256]={0};CK(cuMemcpyHtoD(pdH,z,256*8));CK(cuMemcpyHtoD(pda,z,16*8));
 void*args[]={&pa,&pH,&pdD,&pdH,&pda};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 double gdH[256],gda[16];CK(cuMemcpyDtoH(gdH,pdH,256*8));CK(cuMemcpyDtoH(gda,pda,16*8));
 int fH=0,fA=0;double mH=0,mA=0;
 for(int b=0;b<16;b++)for(int j=0;j<dim;j++){double e=fabs(gdH[b*dim+j]-dH_ref[b*dim+j]);if(e>mH)mH=e;if(e>0.05)fH++;}
 for(int pp=0;pp<dim;pp++){double e=fabs(gda[pp]-da_ref[pp]);if(e>mA)mA=e;if(e>1e-4)fA++;}
 const char*nm=(dim==16)?"SEDENION":"octonion";
 printf("Sounio source-level %s VJP of D=L(a)*H tensor-core GB10:\n",nm);
 printf("  dH = L(a)^T . dD  (tile): mismatch %d/%d  maxerr=%.4f\n",fH,dim*16,mH);
 printf("  da = sum sigma.H.dD (f64): mismatch %d/%d  maxerr=%.2e\n",fA,dim,mA);
 if(!fH&&!fA){printf("PASS: compiler-emitted VJP (dH tile + da accumulation) matches the analytic Jacobian transpose on GB10\n");return 0;}
 printf("FAIL\n");return 1;}

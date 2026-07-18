// Driver-API harness for the compiler-lowered batched hypercomplex ASSOCIATOR tile kernel `step`:
//   out[k*16+b] = [a,b,H_b][k] = ((a⊗b)⊗H_b − a⊗(b⊗H_b))[k]   (k<dim, 16-state batch)
// Usage: run_assoc <ptx> <dim>   (dim=8 octonion / 16 sedenion). Validates vs the scalar CD oracle
// on the DGX Spark GB10. Output is f32 (the wmma tile store format), tolerance covers f16 rounding.
#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
// r = x ⊗ y for a `dim`-dimensional Cayley-Dickson algebra (bits levels).
static void cdmul(const double*x,const double*y,double*r,int dim,int bits){
 for(int k=0;k<dim;k++)r[k]=0;
 for(int i=0;i<dim;i++)for(int j=0;j<dim;j++)r[i^j]+=(double)cds(i,j,bits)*x[i]*y[j];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c,char**v){
 const char*p=c>1?v[1]:"/tmp/assoc.ptx"; int dim=c>2?atoi(v[2]):8; int bits=(dim==16)?4:3;
 double a[16]={0},b[16]={0},H[256]={0};
 for(int i=0;i<dim;i++){a[i]=0.4*((i%2)?-1:1)+0.05*i; b[i]=0.3-0.06*i+((i%3)?0.1:-0.1);}
 for(int s=0;s<16;s++)for(int r=0;r<dim;r++)H[s*dim+r]=((s+1)*0.11+r*0.05)*((r%2)?-1:1);
 // reference assoc[k][s] = ((a⊗b)⊗H_s − a⊗(b⊗H_s))[k]
 double ab[16]; cdmul(a,b,ab,dim,bits);
 double ref[256]={0};
 for(int s=0;s<16;s++){
   const double*Hs=&H[s*dim];
   double left[16],bH[16],right[16];
   cdmul(ab,Hs,left,dim,bits);
   cdmul(b,Hs,bH,dim,bits); cdmul(a,bH,right,dim,bits);
   for(int k=0;k<dim;k++)ref[k*16+s]=left[k]-right[k];
 }
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr pa,pb,pH,po;
 CK(cuMemAlloc(&pa,dim*8));CK(cuMemAlloc(&pb,dim*8));CK(cuMemAlloc(&pH,16*dim*8));CK(cuMemAlloc(&po,256*4));
 CK(cuMemcpyHtoD(pa,a,dim*8));CK(cuMemcpyHtoD(pb,b,dim*8));CK(cuMemcpyHtoD(pH,H,16*dim*8));
 float z[256]={0};CK(cuMemcpyHtoD(po,z,256*4));
 void*args[]={&pa,&pb,&pH,&po};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 float D[256];CK(cuMemcpyDtoH(D,po,256*4));
 int fails=0;double mx=0;int n=dim*16;
 for(int k=0;k<dim;k++)for(int s=0;s<16;s++){double e=fabs(D[k*16+s]-ref[k*16+s]);if(e>mx)mx=e;if(e>0.06)fails++;}
 const char*nm=(dim==16)?"SEDENION":"octonion";
 printf("Sounio source-level BATCHED %s ASSOCIATOR [a,b,H] tensor-core GB10: mismatch %d/%d maxerr=%.4f\n",nm,fails,n,mx);
 if(!fails){printf("PASS: compiler-emitted (a*b)*H - a*(b*H) tensor-core tiles match scalar reference on GB10\n");return 0;}
 printf("FAIL\n");return 1;}

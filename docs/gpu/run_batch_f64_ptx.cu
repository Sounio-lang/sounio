// Validates the batched hypercomplex multiply D = L(a)·H on F64 TENSOR CORES (wmma m8n8k4.f64) against
// the f16→f32 tile (oct_batch_mul) on the DGX Spark GB10 — exact f64 vs the lossy half-precision tile.
// Usage: run_batch_f64 <f64tile.ptx> <f16tile.ptx> <dim>.
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
 const char*p64=c>1?v[1]:"/tmp/f64.ptx";const char*p16=c>2?v[2]:"/tmp/f16.ptx";int DIM=c>3?atoi(v[3]):8;int BITS=(DIM==16)?4:3;
 double a[16]={0},H[256]={0};
 for(int i=0;i<DIM;i++)a[i]=0.4*((i%2)?-1:1)+0.05*i;
 for(int b=0;b<16;b++)for(int r=0;r<DIM;r++)H[b*DIM+r]=((b+1)*0.15+r*0.07)*((r%2)?-1:1);
 double ref[256]={0};
 for(int b=0;b<16;b++)for(int i=0;i<DIM;i++)for(int j=0;j<DIM;j++)ref[(i^j)*16+b]+=(double)cds(i,j,BITS)*a[i]*H[b*DIM+j];
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule m64,m16;CK(cuModuleLoad(&m64,p64));CK(cuModuleLoad(&m16,p16));
 CUfunction f64,f16;CK(cuModuleGetFunction(&f64,m64,"step"));CK(cuModuleGetFunction(&f16,m16,"step"));
 CUdeviceptr pa,pH,po8,po4;CK(cuMemAlloc(&pa,DIM*8));CK(cuMemAlloc(&pH,16*DIM*8));CK(cuMemAlloc(&po8,256*8));CK(cuMemAlloc(&po4,256*4));
 CK(cuMemcpyHtoD(pa,a,DIM*8));CK(cuMemcpyHtoD(pH,H,16*DIM*8));
 double z8[256]={0};CK(cuMemcpyHtoD(po8,z8,256*8));float z4[256]={0};CK(cuMemcpyHtoD(po4,z4,256*4));
 void*a64[]={&pa,&pH,&po8};CK(cuLaunchKernel(f64,1,1,1,32,1,1,0,0,a64,0));CK(cuCtxSynchronize());
 void*a16[]={&pa,&pH,&po4};CK(cuLaunchKernel(f16,1,1,1,32,1,1,0,0,a16,0));CK(cuCtxSynchronize());
 double D8[256];float D4[256];CK(cuMemcpyDtoH(D8,po8,256*8));CK(cuMemcpyDtoH(D4,po4,256*4));
 double m8=0,m4=0;int f8=0;
 for(int k=0;k<DIM;k++)for(int b=0;b<16;b++){
   double e8=fabs(D8[k*16+b]-ref[k*16+b]);if(e8>m8)m8=e8;if(e8>1e-12)f8++;
   double e4=fabs((double)D4[k*16+b]-ref[k*16+b]);if(e4>m4)m4=e4;}
 const char*nm=(DIM==16)?"SEDENION":"octonion";
 printf("Batched %s multiply D = L(a)·H on GB10, maxerr vs host f64:\n",nm);
 printf("  f64 tensor cores (m8n8k4.f64): %.3e  (%d/%d mism)   ← EXACT\n",m8,f8,DIM*16);
 printf("  f16 tile (m16n16k16.f16):      %.3e                ← lossy\n",m4);
 if(f8){printf("FAIL: f64 tensor-core tile not exact\n");return 1;}
 printf("PASS: f64 tensor-core tile is exact to machine precision (vs the f16 tile's ~1e-2..1e-4) on GB10\n");
 return 0;
}

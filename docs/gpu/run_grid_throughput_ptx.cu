// Multi-block throughput of the grid f64 tensor-core tile (oct_batch_mul_f64_grid) on the DGX Spark
// GB10. Each block computes one independent octonion/sedenion batched multiply D = L(a)·H (m8n8k4.f64);
// launching <<<B, 32>>> runs B tiles concurrently. Part 1 checks correctness across blocks; Part 2
// sweeps the block count and reports tiles/s and GFLOP/s. Usage: run_grid_throughput <grid.ptx> <dim>.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c,char**v){
 const char*pg=c>1?v[1]:"/tmp/grid.ptx";int DIM=c>2?atoi(v[2]):8;int BITS=(DIM==16)?4:3;
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext cx;CK(cuDevicePrimaryCtxRetain(&cx,d));CK(cuCtxSetCurrent(cx));
 CUmodule m;CK(cuModuleLoad(&m,pg));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 const int BMAX=131072;
 // per-problem: a[dim], H[16*dim], D[256] f64
 CUdeviceptr da,dH,dD;CK(cuMemAlloc(&da,(size_t)BMAX*DIM*8));CK(cuMemAlloc(&dH,(size_t)BMAX*16*DIM*8));CK(cuMemAlloc(&dD,(size_t)BMAX*256*8));

 // ── Part 1: correctness across blocks ────────────────────────────────────────────────────────
 const int BV=64;
 double *a=(double*)malloc(BV*DIM*8),*H=(double*)malloc((size_t)BV*16*DIM*8);
 for(int b=0;b<BV;b++){for(int i=0;i<DIM;i++)a[b*DIM+i]=0.3*((i%2)?-1:1)+0.05*i+0.01*b;
   for(int st=0;st<16;st++)for(int r=0;r<DIM;r++)H[(size_t)b*16*DIM+st*DIM+r]=0.2*sin(0.4*r+0.6*st+0.1*b);}
 CK(cuMemcpyHtoD(da,a,BV*DIM*8));CK(cuMemcpyHtoD(dH,H,(size_t)BV*16*DIM*8));
 void*args[]={&da,&dH,&dD};CK(cuLaunchKernel(f,BV,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 double*D=(double*)malloc((size_t)BV*256*8);CK(cuMemcpyDtoH(D,dD,(size_t)BV*256*8));
 double mx=0;int fails=0;
 for(int b=0;b<BV;b++)for(int k=0;k<DIM;k++)for(int st=0;st<16;st++){double ref=0;
   for(int i=0;i<DIM;i++)for(int j=0;j<DIM;j++)if((i^j)==k)ref+=(double)cds(i,j,BITS)*a[b*DIM+i]*H[(size_t)b*16*DIM+st*DIM+j];
   double e=fabs(D[(size_t)b*256+k*16+st]-ref);if(e>mx)mx=e;if(e>1e-12)fails++;}
 printf("Part 1 — %d independent blocks (%s), maxerr vs host f64: %.3e  %s\n",BV,(DIM==16)?"sedenion":"octonion",mx,fails?"FAIL":"PASS");
 if(fails)return 1;

 // ── Part 2: throughput sweep ─────────────────────────────────────────────────────────────────
 CUevent e0,e1;CK(cuEventCreate(&e0,0));CK(cuEventCreate(&e1,0));
 int Bs[]={1,64,1024,16384,131072};
 double flop_per_tile=2.0*(double)DIM*DIM*16.0; // 2·dim²·16 MACs*2
 printf("Part 2 — multi-block throughput (GB10, f64 tensor cores m8n8k4):\n");
 printf("   blocks   |   tiles/s   |  GFLOP/s\n");
 for(int bi=0;bi<5;bi++){int B=Bs[bi];
   for(int w=0;w<3;w++){cuLaunchKernel(f,B,1,1,32,1,1,0,0,args,0);}cuCtxSynchronize();
   const int IT=50;CK(cuEventRecord(e0,0));
   for(int it=0;it<IT;it++)cuLaunchKernel(f,B,1,1,32,1,1,0,0,args,0);
   CK(cuEventRecord(e1,0));CK(cuEventSynchronize(e1));
   float ms=0;CK(cuEventElapsedTime(&ms,e0,e1));double s=ms/1e3/IT;
   double tps=B/s, gfps=tps*flop_per_tile/1e9;
   printf("  %8d  | %10.3e | %8.2f\n",B,tps,gfps);
 }
 printf("PASS: grid tile scales across blocks on GB10 (correctness held for %d independent blocks)\n",BV);
 return 0;
}

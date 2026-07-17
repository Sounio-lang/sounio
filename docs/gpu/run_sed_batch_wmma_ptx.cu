#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c,char**v){const char*p=c>1?v[1]:"/tmp/sbw.ptx";
 double a[16],H[256]; for(int i=0;i<16;i++)a[i]=0.4*sin(1.0+i)-0.1*i*0.05;
 for(int b=0;b<16;b++)for(int r=0;r<16;r++)H[b*16+r]=((b+1)*0.11+r*0.037)*((r%2)?-1:1);
 double ref[256]={0};
 for(int b=0;b<16;b++)for(int i=0;i<16;i++)for(int j=0;j<16;j++)ref[(i^j)*16+b]+=(double)cds(i,j,4)*a[i]*H[b*16+j];
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr pa,pH,po;CK(cuMemAlloc(&pa,16*8));CK(cuMemAlloc(&pH,256*8));CK(cuMemAlloc(&po,256*4));
 CK(cuMemcpyHtoD(pa,a,16*8));CK(cuMemcpyHtoD(pH,H,256*8));float z[256]={0};CK(cuMemcpyHtoD(po,z,256*4));
 void*args[]={&pa,&pH,&po};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 float D[256];CK(cuMemcpyDtoH(D,po,256*4));
 int fails=0;double mx=0;for(int k=0;k<16;k++)for(int b=0;b<16;b++){double e=fabs(D[k*16+b]-ref[k*16+b]);if(e>mx)mx=e;if(e>0.05)fails++;}
 printf("Sounio source-level BATCHED sedenion `sed_batch_mul` tensor-core GB10: D=L(a)*H  mismatch %d/256 maxerr=%.4f\n",fails,mx);
 if(!fails){printf("PASS: compiler-emitted tensor-core sedenion L(a)*H tile from source matches scalar reference on GB10\n");return 0;}
 printf("FAIL\n");return 1;}

#include <cstdio>
#include <cmath>
#include <cuda.h>
int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
  int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
  if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
  else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
void om(const float*a,const float*b,float*r){for(int k=0;k<8;k++)r[k]=0;for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=cds(i,j,3)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s L%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c_,char**v){const char*pp=c_>1?v[1]:"/tmp/sounio_ossm_step.ptx";
  FILE*f=fopen(pp,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*ptx=(char*)malloc(n+1);fread(ptx,1,n,f);ptx[n]=0;fclose(f);
  CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuDevicePrimaryCtxRetain(&ctx,d));CK(cuCtxSetCurrent(ctx));
  CUmodule m;CK(cuModuleLoadData(&m,ptx));CUfunction k;CK(cuModuleGetFunction(&k,m,"sounio_ossm_oct_step"));
  float A[8]={0.5f,-0.2f,0.3f,0,0.1f,0,-0.4f,0.2f},B[8]={0.1f,0.2f,-0.1f,0.3f,0,0.1f,0,-0.2f};
  float x[16],H[16*8]; for(int b=0;b<16;b++){x[b]=0.5f+0.1f*b;for(int kk=0;kk<8;kk++)H[b*8+kk]=((b+1)*0.2f+kk*0.1f)*((kk%2)?-1:1);}
  CUdeviceptr dA,dB,dx,dH,dS;CK(cuMemAlloc(&dA,8*4));CK(cuMemAlloc(&dB,8*4));CK(cuMemAlloc(&dx,16*4));CK(cuMemAlloc(&dH,16*8*4));CK(cuMemAlloc(&dS,256*4));
  CK(cuMemcpyHtoD(dA,A,8*4));CK(cuMemcpyHtoD(dB,B,8*4));CK(cuMemcpyHtoD(dx,x,16*4));CK(cuMemcpyHtoD(dH,H,16*8*4));
  void*ar[]={&dA,&dB,&dx,&dH,&dS}; CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());
  float hS[256]; CK(cuMemcpyDtoH(hS,dS,256*4));
  int fails=0;float mx=0; for(int b=0;b<16;b++){float ah[8];om(A,&H[b*8],ah);for(int kk=0;kk<8;kk++){float ref=ah[kk]+B[kk]*x[b];float e=fabsf(hS[kk*16+b]-ref);if(e>mx)mx=e;if(e>0.05f)fails++;}}
  printf("Sounio O-SSM step PTX GB10: S=A⊗H+B·x  batch %d/128 mismatch maxerr=%.4f\n",fails,mx);
  if(!fails){printf("PASS: Sounio-emitted tensor-core O-SSM octonion step matches scalar reference on GB10\n");return 0;}
  printf("FAIL\n");return 1;}

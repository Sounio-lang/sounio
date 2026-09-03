#include <cstdio>
#include <cmath>
#include <cuda.h>
int cd_sigma(int a,int b,int bits){ if(a==0||b==0)return 1; if(bits<=1)return -1;
  int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
  if(!ah&&!bh)return cd_sigma(al,bl,bits-1); if(!ah&&bh)return cd_sigma(bl,al,bits-1);
  if(ah&&!bh)return bl==0?cd_sigma(al,0,bits-1):-cd_sigma(al,bl,bits-1);
  if(bl==0)return -cd_sigma(0,al,bits-1); return cd_sigma(bl,al,bits-1); }
void ref(const float*a,const float*b,float*r){for(int k=0;k<8;k++)r[k]=0;
  for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=cd_sigma(i,j,3)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s L%d: %s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c_,char**v){ const char*pp=c_>1?v[1]:"/tmp/sounio_oct_full.ptx";
  FILE*f=fopen(pp,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*ptx=(char*)malloc(n+1);fread(ptx,1,n,f);ptx[n]=0;fclose(f);
  CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuDevicePrimaryCtxRetain(&ctx,d));CK(cuCtxSetCurrent(ctx));
  CUmodule m;CK(cuModuleLoadData(&m,ptx));CUfunction k;CK(cuModuleGetFunction(&k,m,"sounio_oct_mul_full"));
  float a[8]={1,2,0,-1,3,0,1,-2}; float B[16*8]; for(int cc=0;cc<16;cc++)for(int j=0;j<8;j++)B[cc*8+j]=((cc+1)*0.5f+j*0.25f)*((j%2)?-1:1);
  float hD[256]; CUdeviceptr da,db,dd; CK(cuMemAlloc(&da,8*4));CK(cuMemAlloc(&db,16*8*4));CK(cuMemAlloc(&dd,256*4));
  CK(cuMemcpyHtoD(da,a,8*4));CK(cuMemcpyHtoD(db,B,16*8*4)); void*ar[]={&da,&db,&dd};
  CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(hD,dd,256*4));
  int fails=0;float mx=0; for(int cc=0;cc<16;cc++){float rr[8];ref(a,&B[cc*8],rr);for(int kk=0;kk<8;kk++){float er=fabsf(hD[kk*16+cc]-rr[kk]);if(er>mx)mx=er;if(er>0.15f){fails++;if(fails<=4)printf("  c=%d k=%d got=%.3f ref=%.3f\n",cc,kk,hD[kk*16+cc],rr[kk]);}}}
  float a2[8]={0,1,0,0,0,0,0,0},B2[16*8]={0};B2[2]=1.0f; CK(cuMemcpyHtoD(da,a2,8*4));CK(cuMemcpyHtoD(db,B2,16*8*4));
  CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(hD,dd,256*4));
  printf("in-kernel Sounio PTX GB10: e1*e2 comp3=%.2f comp4=%.2f | batch %d/128 maxerr=%.3f\n",hD[3*16+0],hD[4*16+0],fails,mx);
  if(!fails&&hD[3*16+0]>0.7f&&hD[4*16+0]<0.3f){printf("PASS: in-kernel Sounio WMMA octonion multiply is Convention X on GB10\n");return 0;}
  printf("FAIL\n");return 1; }

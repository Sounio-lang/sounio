#include <cstdio>
#include <cmath>
#include <cuda.h>
int cd_sigma(int a,int b,int bits){ if(a==0||b==0)return 1; if(bits<=1)return -1;
  int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
  if(!ah&&!bh)return cd_sigma(al,bl,bits-1); if(!ah&&bh)return cd_sigma(bl,al,bits-1);
  if(ah&&!bh)return bl==0?cd_sigma(al,0,bits-1):-cd_sigma(al,bl,bits-1);
  if(bl==0)return -cd_sigma(0,al,bits-1); return cd_sigma(bl,al,bits-1); }
void ref(const float*a,const float*b,float*r){for(int k=0;k<16;k++)r[k]=0;
  for(int i=0;i<16;i++)for(int j=0;j<16;j++)r[i^j]+=cd_sigma(i,j,4)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s L%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c_,char**v){ const char*pp=c_>1?v[1]:"/tmp/sounio_sed_full.ptx";
  FILE*f=fopen(pp,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*ptx=(char*)malloc(n+1);fread(ptx,1,n,f);ptx[n]=0;fclose(f);
  CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuDevicePrimaryCtxRetain(&ctx,d));CK(cuCtxSetCurrent(ctx));
  CUmodule m;CK(cuModuleLoadData(&m,ptx));CUfunction k;CK(cuModuleGetFunction(&k,m,"sounio_sed_mul_full"));
  CUdeviceptr da,db,dd; CK(cuMemAlloc(&da,16*4));CK(cuMemAlloc(&db,256*4));CK(cuMemAlloc(&dd,256*4)); void*ar[]={&da,&db,&dd}; float hD[256];
  float a[16]; for(int i=0;i<16;i++)a[i]=(i%3)-1+0.5f*(i%2?-1:1);
  float B[256]; for(int cc=0;cc<16;cc++)for(int r=0;r<16;r++)B[cc*16+r]=((cc+1)*0.25f+r*0.125f)*((r%2)?-1:1);
  CK(cuMemcpyHtoD(da,a,16*4));CK(cuMemcpyHtoD(db,B,256*4)); CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(hD,dd,256*4));
  int fails=0;float mx=0; for(int cc=0;cc<16;cc++){float rr[16];ref(a,&B[cc*16],rr);for(int kk=0;kk<16;kk++){float er=fabsf(hD[kk*16+cc]-rr[kk]);if(er>mx)mx=er;if(er>0.3f)fails++;}}
  float a2[16]={0};a2[2]=1; float B2[256]={0};B2[5]=1; CK(cuMemcpyHtoD(da,a2,16*4));CK(cuMemcpyHtoD(db,B2,256*4));
  CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(hD,dd,256*4)); float e25=hD[7*16+0];
  float aZ[16]={0};aZ[3]=1;aZ[10]=1; float BZ[256]={0};BZ[6]=1;BZ[15]=-1; CK(cuMemcpyHtoD(da,aZ,16*4));CK(cuMemcpyHtoD(db,BZ,256*4));
  CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(hD,dd,256*4));
  int znz=0;float zmax=0; for(int kk=0;kk<16;kk++){float vv=fabsf(hD[kk*16+0]);if(vv>zmax)zmax=vv;if(vv>0.05f)znz++;}
  printf("SEDENION Sounio PTX GB10: e2*e5->comp7=%.2f | batch %d/256 maxerr=%.3f | ZD nonzero=%d/16 max=%.4f\n",e25,fails,mx,znz,zmax);
  if(!fails&&e25>0.7f&&!znz){printf("PASS: Sounio-emitted tensor-core sedenion multiply is Convention X + ZD annihilates on GB10\n");return 0;}
  printf("FAIL\n");return 1; }

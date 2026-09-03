#include <cstdio>
#include <cmath>
#include <cuda.h>
int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
  int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
  if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
  else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
void om(const float*a,const float*b,float*r){for(int k=0;k<8;k++)r[k]=0;for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=cds(i,j,3)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s L%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c_,char**v){const char*pp=c_>1?v[1]:"/tmp/sounio_assoc.ptx";
  FILE*f=fopen(pp,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*ptx=(char*)malloc(n+1);fread(ptx,1,n,f);ptx[n]=0;fclose(f);
  CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuDevicePrimaryCtxRetain(&ctx,d));CK(cuCtxSetCurrent(ctx));
  CUmodule m;CK(cuModuleLoadData(&m,ptx));CUfunction k;CK(cuModuleGetFunction(&k,m,"sounio_oct_associator"));
  CUdeviceptr da,dc,dM;CK(cuMemAlloc(&da,8*4));CK(cuMemAlloc(&dc,8*4));CK(cuMemAlloc(&dM,256*4));void*ar[]={&da,&dc,&dM};float hM[256];
  float mx=0;unsigned s=999;auto rn=[&](){s=s*1103515245+12345;return((s>>16)&0x7fff)/16384.0f-1.0f;};
  for(int t=0;t<50;t++){float a[8],b[8],c[8];for(int i=0;i<8;i++){a[i]=rn();b[i]=rn();c[i]=rn();}
    CK(cuMemcpyHtoD(da,a,8*4));CK(cuMemcpyHtoD(dc,c,8*4));CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(hM,dM,256*4));
    float ab[8],bc[8],p1[8],p2[8];om(a,b,ab);om(ab,c,p1);om(b,c,bc);om(a,bc,p2);
    for(int kk=0;kk<8;kk++){float Mb=0;for(int j=0;j<8;j++)Mb+=hM[kk*16+j]*b[j];float e=fabsf(Mb-(p1[kk]-p2[kk]));if(e>mx)mx=e;}}
  int cnt=0,n0=0,n2=0,oth=0;
  for(int i=1;i<8;i++)for(int kk=1;kk<8;kk++){float a[8]={0},c[8]={0};a[i]=1;c[kk]=1;
    CK(cuMemcpyHtoD(da,a,8*4));CK(cuMemcpyHtoD(dc,c,8*4));CK(cuLaunchKernel(k,1,1,1,32,1,1,0,0,ar,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(hM,dM,256*4));
    for(int j=1;j<8;j++){float q=0;for(int r=0;r<8;r++){float vv=hM[r*16+j];q+=vv*vv;}if(q>0.5f)cnt++;float nr=sqrtf(q);if(nr<0.3f)n0++;else if(fabsf(nr-2)<0.3f)n2++;else oth++;}}
  printf("Sounio associator PTX GB10: M·b vs direct max=%.4f | non-assoc triples=%d (exp 168) | norms 0:%d 2:%d other:%d\n",mx,cnt,n0,n2,oth);
  if(mx<0.15f&&cnt==168&&oth==0){printf("PASS: Sounio-emitted tensor-core octonion associator — 168 + dichotomy {0,2} on GB10\n");return 0;}
  printf("FAIL\n");return 1;}

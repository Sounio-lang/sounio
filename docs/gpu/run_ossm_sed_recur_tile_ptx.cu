#include <cstdio>
#include <cmath>
#include <cuda.h>
#define T 6
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
static float sigc(float x){return 0.015625f*x*x*x+0.25f*x+0.5f;}
static void sm(const double*a,const double*b,double*r){for(int k=0;k<16;k++)r[k]=0;for(int i=0;i<16;i++)for(int j=0;j<16;j++)r[i^j]+=(double)cds(i,j,4)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c,char**v){const char*p=c>1?v[1]:"/tmp/sedr.ptx";
 double A[16],Bb[16],C[16],xx[256]={0},H0[256];
 for(int i=0;i<16;i++){A[i]=0.4*sin(1.0+i);Bb[i]=0.15*cos(0.5+i);C[i]=0.2*sin(2.0+i);}
 for(int t=0;t<T;t++)for(int b=0;b<16;b++)xx[t*16+b]=0.3+0.04*b-0.015*t;
 for(int b=0;b<16;b++)for(int r=0;r<16;r++)H0[b*16+r]=((b+1)*0.09+r*0.03)*((r%2)?-1:1);
 double h[16][16]; for(int b=0;b<16;b++)for(int k=0;k<16;k++)h[b][k]=H0[b*16+k];
 double yref[256]={0};
 for(int t=0;t<T;t++)for(int b=0;b<16;b++){double ah[16];sm(A,h[b],ah);double hn[16];
   for(int k=0;k<16;k++)hn[k]=sigc((float)(ah[k]+Bb[k]*xx[t*16+b]));
   for(int k=0;k<16;k++)h[b][k]=hn[k];
   double ch[16];sm(C,h[b],ch);yref[t*16+b]=ch[0];}
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext X;CK(cuDevicePrimaryCtxRetain(&X,d));CK(cuCtxSetCurrent(X));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr pA,pB,px,pC,pH,py;CK(cuMemAlloc(&pA,16*8));CK(cuMemAlloc(&pB,16*8));CK(cuMemAlloc(&px,256*8));CK(cuMemAlloc(&pC,16*8));CK(cuMemAlloc(&pH,256*8));CK(cuMemAlloc(&py,256*4));
 CK(cuMemcpyHtoD(pA,A,16*8));CK(cuMemcpyHtoD(pB,Bb,16*8));CK(cuMemcpyHtoD(px,xx,256*8));CK(cuMemcpyHtoD(pC,C,16*8));CK(cuMemcpyHtoD(pH,H0,256*8));float z[256]={0};CK(cuMemcpyHtoD(py,z,256*4));
 long long Tv=T;void*args[]={&pA,&pB,&px,&pC,&pH,&py,&Tv};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 float Y[256];CK(cuMemcpyDtoH(Y,py,256*4));
 int fails=0;double mx=0;for(int t=0;t<T;t++)for(int b=0;b<16;b++){double e=fabs(Y[t*16+b]-yref[t*16+b]);if(e>mx)mx=e;if(e>0.04)fails++;}
 printf("Sounio source-level O-SSM SEDENION RECURRENCE (T=%d) tensor-core GB10: y mismatch %d/%d maxerr=%.4f\n",T,fails,T*16,mx);
 if(!fails){printf("PASS: compiler-lowered O-SSM sedenion T-step recurrence matches scalar reference on GB10\n");return 0;}
 printf("FAIL\n");return 1;}

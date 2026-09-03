#include <cstdio>
#include <cmath>
#include <cuda.h>
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
static float sigc(float x){return 0.015625f*x*x*x+0.25f*x+0.5f;}
static void om(const double*a,const double*b,double*r){for(int k=0;k<8;k++)r[k]=0;for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=(double)cds(i,j,3)*a[i]*b[j];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
int main(int c,char**v){const char*p=c>1?v[1]:"/tmp/oct.ptx";
 double A[8]={0.5,-0.2,0.3,0.0,0.1,0.0,-0.4,0.2}, B[8]={0.1,0.2,-0.1,0.3,0.0,0.1,0.0,-0.2};
 double C[8]={0.3,0.1,-0.2,0.15,-0.05,0.2,0.1,-0.1};
 double xx[16],H[128]; for(int b=0;b<16;b++){xx[b]=0.3+0.05*b; for(int r=0;r<8;r++)H[b*8+r]=((b+1)*0.15+r*0.07)*((r%2)?-1:1);}
 double yref[16];
 for(int b=0;b<16;b++){ double ah[8]; om(A,&H[b*8],ah); double h[8];
   for(int k=0;k<8;k++) h[k]=sigc((float)(ah[k]+B[k]*xx[b]));
   double ch[8]; om(C,h,ch); yref[b]=ch[0]; }
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext X;CK(cuDevicePrimaryCtxRetain(&X,d));CK(cuCtxSetCurrent(X));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr pA,pB,px,pC,pH,py;CK(cuMemAlloc(&pA,8*8));CK(cuMemAlloc(&pB,8*8));CK(cuMemAlloc(&px,16*8));CK(cuMemAlloc(&pC,8*8));CK(cuMemAlloc(&pH,128*8));CK(cuMemAlloc(&py,16*4));
 CK(cuMemcpyHtoD(pA,A,8*8));CK(cuMemcpyHtoD(pB,B,8*8));CK(cuMemcpyHtoD(px,xx,16*8));CK(cuMemcpyHtoD(pC,C,8*8));CK(cuMemcpyHtoD(pH,H,128*8));float z[16]={0};CK(cuMemcpyHtoD(py,z,16*4));
 void*args[]={&pA,&pB,&px,&pC,&pH,&py};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 float Y[16];CK(cuMemcpyDtoH(Y,py,16*4));
 int fails=0;double mx=0;for(int b=0;b<16;b++){double e=fabs(Y[b]-yref[b]);if(e>mx)mx=e;if(e>0.02)fails++;}
 printf("Sounio source-level FULL O-SSM cell y=Re(C@sigmoid(A@H+B.x)) tensor-core GB10: mismatch %d/16 maxerr=%.4f\n",fails,mx);
 if(!fails){printf("PASS: compiler-lowered full O-SSM tensor-core cell matches scalar reference on GB10\n");return 0;}
 printf("FAIL\n");return 1;}

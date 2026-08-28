// Validates the hypercomplex MATRIX-vector (real-capacity) kernel H_out[i]=Σ_j A[i][j]⊗H[j] over D_h
// octonion channels on the DGX Spark GB10. Packed as A_big(N×N)·H_big(N×16) f16 m16n16k16 tiles, N=D_h·8.
// Usage: run_matvec <ptx> <D_h>   (8 or 16). Output is f32 D-layout [comp*16+batch]; tolerance covers f16.
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
 const char*p=c>1?v[1]:"/tmp/mv.ptx"; int DH=c>2?atoi(v[2]):16; int N=DH*8;
 int na=DH*DH*8, nh=DH*8*16, no=N*16;
 double *A=(double*)malloc(na*8), *H=(double*)malloc(nh*8);
 for(int i=0;i<DH;i++)for(int j=0;j<DH;j++)for(int m=0;m<8;m++)A[i*N+j*8+m]=0.15*sin(0.3*i+0.5*j+0.7*m)*((m%2)?-1:1);
 for(int b=0;b<16;b++)for(int j=0;j<DH;j++)for(int l=0;l<8;l++)H[b*N+j*8+l]=0.2*cos(0.4*j+0.6*l+0.2*b);
 // host ref: out[(i*8+k)*16+b] = Σ_{j,l} cds(k⊕l,l)·A[i*N+j*8+(k⊕l)]·H[b*N+j*8+l]
 double *ref=(double*)malloc(no*8); double mref=0;
 for(int i=0;i<DH;i++)for(int k=0;k<8;k++)for(int b=0;b<16;b++){double s=0;
   for(int j=0;j<DH;j++)for(int l=0;l<8;l++){int m=k^l;s+=(double)cds(m,l,3)*A[i*N+j*8+m]*H[b*N+j*8+l];}
   ref[(i*8+k)*16+b]=s; if(fabs(s)>mref)mref=fabs(s);}
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr da,dh,dp;CK(cuMemAlloc(&da,na*8));CK(cuMemAlloc(&dh,nh*8));CK(cuMemAlloc(&dp,no*4));
 CK(cuMemcpyHtoD(da,A,na*8));CK(cuMemcpyHtoD(dh,H,nh*8));float*z=(float*)calloc(no,4);CK(cuMemcpyHtoD(dp,z,no*4));
 void*args[]={&da,&dh,&dp};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
 float*D=(float*)malloc(no*4);CK(cuMemcpyDtoH(D,dp,no*4));
 double mx=0;int fails=0;
 for(int i=0;i<no;i++){double e=fabs((double)D[i]-ref[i]);if(e>mx)mx=e;}
 double tol=0.03*mref+0.02;
 for(int i=0;i<no;i++)if(fabs((double)D[i]-ref[i])>tol)fails++;
 printf("Hypercomplex matrix-vector H_out=A⊛H, D_h=%d (%d octonion weights, A_big %dx%d, %d m16n16k16 tiles) on GB10:\n",
        DH,DH*DH,N,N,(N/16)*(N/16));
 printf("  mismatch %d/%d  maxerr=%.4f  (|ref|max=%.3f, f16 tile tol=%.3f)\n",fails,no,mx,mref,tol);
 if(!fails){printf("PASS: real-capacity hypercomplex matrix-vector fills the tensor cores and matches the scalar reference (f16) on GB10\n");return 0;}
 printf("FAIL\n");return 1;
}

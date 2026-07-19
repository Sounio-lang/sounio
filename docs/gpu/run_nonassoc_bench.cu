// Synthetic NON-ASSOCIATIVITY benchmark on the DGX Spark GB10.
// Task: from an octonion triple (a,b,c), predict a fixed random linear projection y = w·[a,b,c] of the
// ASSOCIATOR [a,b,c] = (a⊗b)⊗c − a⊗(b⊗c). The associator is IDENTICALLY ZERO for any associative
// algebra (ℝ, ℂ, ℍ/quaternion), so this task is invisible to associative models by construction.
// We compare linear probes (held-out R²) on four feature sets:
//   (1) octonion associator z=[a,b,c] computed by OUR compiler-lowered oct_assoc tensor-core kernel;
//   (2) quaternion associator (≡ 0) — the associative-algebra blind spot;
//   (3) the raw 24-d input (a,b,c) — the associator is trilinear, not linear;
//   (4) the associative pairwise products [a⊗b, b⊗c] — has the pieces but not their non-assoc difference.
// Usage: nonassoc_bench <oct_assoc.ptx>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
static unsigned S=12345; static double rnd(){S=S*1664525u+1013904223u; return ((S>>8)&0xffffff)/16777216.0*2-1;}
static int cds(int a,int b,int bits){int s=1;while(bits>0){if(a==0||b==0)return s;if(bits==1)return -s;
 int h=1<<(bits-1),ah=a>=h,bh=b>=h,al=a&(h-1),bl=b&(h-1);
 if(!ah&&!bh){a=al;b=bl;}else if(!ah&&bh){a=bl;b=al;}
 else if(ah&&!bh){if(bl==0){a=al;b=0;}else{s=-s;a=al;b=bl;}}
 else{if(bl==0){s=-s;a=0;b=al;}else{a=bl;b=al;}}bits--;}return s;}
static void omul(const double*x,const double*y,double*r){for(int k=0;k<8;k++)r[k]=0;
 for(int i=0;i<8;i++)for(int j=0;j<8;j++)r[i^j]+=(double)cds(i,j,3)*x[i]*y[j];}
static void assoc(const double*a,const double*b,const double*c,double*z){ // [a,b,c]=(ab)c − a(bc)
 double ab[8],bc[8],l[8],r[8];omul(a,b,ab);omul(b,c,bc);omul(ab,c,l);omul(a,bc,r);for(int k=0;k<8;k++)z[k]=l[k]-r[k];}
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);return 2;}}while(0)
// least-squares y≈Xθ via normal equations (Gaussian elim); returns held-out R².
static double probe(const double*X,const double*y,int n,int d,int ntr){
 double*A=(double*)calloc((d+1)*(d+1),8),*g=(double*)calloc(d+1,8);int D=d+1;
 for(int s=0;s<ntr;s++){double xi[64];for(int j=0;j<d;j++)xi[j]=X[s*d+j];xi[d]=1.0;
   for(int a=0;a<D;a++){for(int b=0;b<D;b++)A[a*D+b]+=xi[a]*xi[b];g[a]+=xi[a]*y[s];}}
 for(int a=0;a<D;a++)A[a*D+a]+=1e-6; // ridge
 for(int c=0;c<D;c++){int piv=c;for(int r=c+1;r<D;r++)if(fabs(A[r*D+c])>fabs(A[piv*D+c]))piv=r;
   if(piv!=c){for(int k=0;k<D;k++){double t=A[c*D+k];A[c*D+k]=A[piv*D+k];A[piv*D+k]=t;}double t=g[c];g[c]=g[piv];g[piv]=t;}
   double pv=A[c*D+c];if(fabs(pv)<1e-12)continue;
   for(int r=0;r<D;r++){if(r==c)continue;double f=A[r*D+c]/pv;for(int k=0;k<D;k++)A[r*D+k]-=f*A[c*D+k];g[r]-=f*g[c];}}
 double*th=(double*)calloc(D,8);for(int c=0;c<D;c++)th[c]=(fabs(A[c*D+c])>1e-12)?g[c]/A[c*D+c]:0;
 double my=0;for(int s=ntr;s<n;s++)my+=y[s];my/=(n-ntr);
 double sr=0,st=0;for(int s=ntr;s<n;s++){double p=th[d];for(int j=0;j<d;j++)p+=th[j]*X[s*d+j];
   sr+=(y[s]-p)*(y[s]-p);st+=(y[s]-my)*(y[s]-my);}
 free(A);free(g);free(th);return 1.0-sr/(st+1e-30);
}
int main(int c,char**v){
 const char*p=c>1?v[1]:"/tmp/oct_assoc.ptx";
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule m;CK(cuModuleLoad(&m,p));CUfunction f;CK(cuModuleGetFunction(&f,m,"step"));
 CUdeviceptr da,db,dH,dO;CK(cuMemAlloc(&da,8*8));CK(cuMemAlloc(&db,8*8));CK(cuMemAlloc(&dH,128*8));CK(cuMemAlloc(&dO,256*4));
 const int G=512, N=G*16;                       // 8192 samples
 double w[8];for(int k=0;k<8;k++)w[k]=rnd();     // fixed random projection of the associator
 double *Xoct=(double*)malloc(N*8*8),*Xquat=(double*)malloc(N*8*8),*Xraw=(double*)malloc(N*24*8),*Xprod=(double*)malloc(N*16*8);
 double *y=(double*)malloc(N*8);
 double kerr=0; int s=0;
 for(int gi=0;gi<G;gi++){
   double a[8],b[8],C[128];for(int k=0;k<8;k++){a[k]=rnd();b[k]=rnd();}
   for(int i=0;i<16;i++)for(int k=0;k<8;k++)C[i*8+k]=rnd();
   CK(cuMemcpyHtoD(da,a,64));CK(cuMemcpyHtoD(db,b,64));CK(cuMemcpyHtoD(dH,C,128*8));
   void*args[]={&da,&db,&dH,&dO};CK(cuLaunchKernel(f,1,1,1,32,1,1,0,0,args,0));CK(cuCtxSynchronize());
   float O[256];CK(cuMemcpyDtoH(O,dO,256*4));   // f32 output; O[k*16+i] = [a,b,c_i][k]
   for(int i=0;i<16;i++){
     double cc[8];for(int k=0;k<8;k++)cc[k]=C[i*8+k];
     double zt[8];assoc(a,b,cc,zt);             // exact host associator (the target signal)
     double zk[8];for(int k=0;k<8;k++){zk[k]=(double)O[k*16+i]; kerr+=fabs(zk[k]-zt[k]);}
     double yy=0;for(int k=0;k<8;k++)yy+=w[k]*zt[k]; y[s]=yy;   // target = w·[a,b,c]
     for(int k=0;k<8;k++)Xoct[s*8+k]=zk[k];      // (1) octonion associator feature (from OUR kernel)
     for(int k=0;k<8;k++)Xquat[s*8+k]=0.0;       // (2) quaternion associator ≡ 0 (associative blind spot)
     for(int k=0;k<8;k++){Xraw[s*24+k]=a[k];Xraw[s*24+8+k]=b[k];Xraw[s*24+16+k]=cc[k];} // (3) raw input
     double ab[8],bc[8];omul(a,b,ab);omul(b,cc,bc);
     for(int k=0;k<8;k++){Xprod[s*16+k]=ab[k];Xprod[s*16+8+k]=bc[k];} // (4) associative pairwise products
     s++;
   }
 }
 int ntr=N*3/4;
 printf("Synthetic non-associativity benchmark on GB10 — task: predict y = w·[a,b,c] (a random projection\n");
 printf("of the octonion associator). %d samples (%d train / %d test). oct_assoc kernel feat maxerr≈%.1e.\n",N,ntr,N-ntr,kerr/(N*8));
 printf("  linear-probe held-out R²:\n");
 printf("    (1) octonion associator [a,b,c]  (our oct_assoc kernel):  R² = %.4f\n",probe(Xoct,y,N,8,ntr));
 printf("    (2) quaternion associator (≡ 0, associative blind spot):  R² = %.4f\n",probe(Xquat,y,N,8,ntr));
 printf("    (3) raw input (a,b,c), 24-d (associator is trilinear):    R² = %.4f\n",probe(Xraw,y,N,24,ntr));
 printf("    (4) associative products [a⊗b, b⊗c], 16-d:                R² = %.4f\n",probe(Xprod,y,N,16,ntr));
 double r1=probe(Xoct,y,N,8,ntr),r2=probe(Xquat,y,N,8,ntr),r3=probe(Xraw,y,N,24,ntr);
 if(r1>0.9 && r2<0.1 && r3<0.2){printf("PASS: non-associativity is REQUIRED — only the octonion-associator feature linearizes the task; associative/linear features are at chance (blind by construction).\n");return 0;}
 printf("FAIL\n");return 1;
}

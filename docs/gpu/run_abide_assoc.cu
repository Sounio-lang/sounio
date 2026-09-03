// REAL DATA: ABIDE-I ASD/TD classification, honest head-to-head. Does the octonion associator of the
// connectome sequence carry ASD/TD signal that associative features miss? Each subject is 8 octonions
// h_0..h_7 (the brain_ossm.abide.v2 8×8 feature layout, 250 ASD / 250 TD, 20 sites). The octonion
// associator features [h_t,h_{t+1},h_{t+2}] (t=0..5) are computed on the compiler-lowered tensor-core
// kernel (oct_assoc) on the GB10 — the SAME operation validated bit-for-bit in #1204/#1212. The
// classifier is L2 logistic regression under LEAVE-ONE-SITE-OUT cross-validation (site is a known
// confound — see the G2 null). Balanced accuracy reported per model. Honest: prior octonion methods on
// ABIDE ASD/TD were at chance (G2-Gram d=0.06; O-SSM 49.5%); this tests the edge/sequence-associator
// door with the trainable/validated kernel and proper CV.
// Usage: run_abide_assoc <oct_assoc.ptx> <manifest.tsv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>
#define CK(x) do{CUresult e=(x);if(e){const char*s;cuGetErrorString(e,&s);printf("ERR %s@%d:%s\n",#x,__LINE__,s);exit(2);}}while(0)
static const int NS=500, NSITE=20, T=8, FEAT=64;
static CUfunction ASS; static CUdeviceptr da_,db_,dH_,dO_;
// oct_assoc(a,b,H) → D-layout f32; slot0 of the batch holds the associator [a,b,H0]
static void assoc(const double*a,const double*b,const double*c,double*out8){
 double C[128]={0}; for(int k=0;k<8;k++)C[k]=c[k];        // batch slot 0 = c
 cuMemcpyHtoD(da_,a,64);cuMemcpyHtoD(db_,b,64);cuMemcpyHtoD(dH_,C,128*8);float zz[256]={0};cuMemcpyHtoD(dO_,zz,256*4);
 void*A[]={&da_,&db_,&dH_,&dO_};cuLaunchKernel(ASS,1,1,1,32,1,1,0,0,A,0);cuCtxSynchronize();
 float z[256];cuMemcpyDtoH(z,dO_,256*4); for(int k=0;k<8;k++)out8[k]=z[k*16+0];   // D-layout [comp*16+batch]
}
static int lab[NS], site[NS]; static char sname[NSITE][32]; static int nsite=0;
static double feat[NS][FEAT];
static int site_idx(const char*s){for(int i=0;i<nsite;i++)if(!strcmp(sname[i],s))return i;strcpy(sname[nsite],s);return nsite++;}
// balanced-accuracy L2 logistic regression, GD; z-score by TRAIN stats; returns test balanced acc
static double loso_fold(double*X,int D,int held){
 double mu[512]={0},sd[512]={0};int ntr=0;
 for(int i=0;i<NS;i++)if(site[i]!=held){ntr++;for(int j=0;j<D;j++)mu[j]+=X[i*D+j];}
 for(int j=0;j<D;j++)mu[j]/=ntr;
 for(int i=0;i<NS;i++)if(site[i]!=held)for(int j=0;j<D;j++){double d=X[i*D+j]-mu[j];sd[j]+=d*d;}
 for(int j=0;j<D;j++){sd[j]=sqrt(sd[j]/ntr);if(sd[j]<1e-9)sd[j]=1;}
 double w[512]={0},b=0; const double lr=0.2,l2=0.02; const int IT=500;
 for(int it=0;it<IT;it++){double gw[512]={0},gb=0;
  for(int i=0;i<NS;i++)if(site[i]!=held){double z=b;for(int j=0;j<D;j++)z+=w[j]*(X[i*D+j]-mu[j])/sd[j];
    double p=1.0/(1.0+exp(-z)),g=p-lab[i];for(int j=0;j<D;j++)gw[j]+=g*(X[i*D+j]-mu[j])/sd[j];gb+=g;}
  for(int j=0;j<D;j++)w[j]-=lr*(gw[j]/ntr+l2*w[j]);b-=lr*gb/ntr;}
 int tp=0,tn=0,np=0,nn=0;
 for(int i=0;i<NS;i++)if(site[i]==held){double z=b;for(int j=0;j<D;j++)z+=w[j]*(X[i*D+j]-mu[j])/sd[j];
   int pr=z>0; if(lab[i]){np++;if(pr)tp++;}else{nn++;if(!pr)tn++;}}
 double sens=np?(double)tp/np:0.5, spec=nn?(double)tn/nn:0.5; return 0.5*(sens+spec);
}
static void report(const char*name,double*X,int D){
 double s=0,s2=0;int n=0;
 for(int h=0;h<nsite;h++){double a=loso_fold(X,D,h);s+=a;s2+=a*a;n++;}
 double m=s/n,sd=sqrt(s2/n-m*m);
 printf("  %-34s LOSO bal-acc = %5.1f%% ± %4.1f  (%d sites)\n",name,100*m,100*sd,n);
}
int main(int c,char**v){
 const char*pa=c>1?v[1]:"/tmp/tr_assoc.ptx"; const char*mf=c>2?v[2]:"/tmp/abide_manifest.tsv";
 FILE*f=fopen(mf,"r"); if(!f){printf("no manifest %s\n",mf);return 2;} char line[8192]; int row=0;
 while(fgets(line,sizeof line,f)){ if(line[0]=='#')continue;
   char*t=strtok(line,"\t"); if(!t)continue; if(!strcmp(t,"subject_id"))continue;  // header
   if(row>=NS)break; char*ls=strtok(0,"\t"),*ss=strtok(0,"\t");
   lab[row]= (ls&&ls[0]=='A')?1:0;                          // ASD=1, TD=0
   site[row]=site_idx(ss);
   for(int j=0;j<FEAT;j++){char*fs=strtok(0,"\t \n"); feat[row][j]=fs?atof(fs):0;}
   row++;
 } fclose(f);
 printf("Loaded %d subjects, %d sites (ASD=%d).\n",row,nsite,[&]{int a=0;for(int i=0;i<row;i++)a+=lab[i];return a;}());
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuDevicePrimaryCtxRetain(&x,d));CK(cuCtxSetCurrent(x));
 CUmodule ma;CK(cuModuleLoad(&ma,pa));CK(cuModuleGetFunction(&ASS,ma,"step"));
 CK(cuMemAlloc(&da_,64));CK(cuMemAlloc(&db_,64));CK(cuMemAlloc(&dH_,128*8));CK(cuMemAlloc(&dO_,256*4));
 // ── feature builds ──
 static double Xraw[NS*64], Xoct[NS*54], Xquat[NS*54], Xcat[NS*(64+54)];
 for(int i=0;i<NS;i++){
  for(int j=0;j<64;j++)Xraw[i*64+j]=feat[i][j];
  double h[8][8]; for(int t=0;t<8;t++)for(int k=0;k<8;k++)h[t][k]=feat[i][t*8+k];
  int c54=0;
  for(int t=0;t<6;t++){double z[8]; assoc(h[t],h[t+1],h[t+2],z);
    double nrm=0; for(int k=0;k<8;k++){Xoct[i*54+c54++]=z[k];nrm+=z[k]*z[k];} Xoct[i*54+c54++]=sqrt(nrm);}
  // quaternion associator: truncate to comps 0..3 (a,b,c), quaternion assoc ≡ 0 → feature ≈ 0
  int q54=0;
  for(int t=0;t<6;t++){double aq[8]={0},bq[8]={0},cq[8]={0}; for(int k=0;k<4;k++){aq[k]=h[t][k];bq[k]=h[t+1][k];cq[k]=h[t+2][k];}
    double z[8]; assoc(aq,bq,cq,z); double nrm=0; for(int k=0;k<8;k++){Xquat[i*54+q54++]=z[k];nrm+=z[k]*z[k];} Xquat[i*54+q54++]=sqrt(nrm);}
  for(int j=0;j<64;j++)Xcat[i*118+j]=Xraw[i*64+j]; for(int j=0;j<54;j++)Xcat[i*118+64+j]=Xoct[i*54+j];
 }
 printf("ABIDE-I ASD/TD, leave-one-site-out CV (site = confound). Chance = 50.0%%. Associator features from oct_assoc on GB10.\n");
 report("RAW 8x8 features (associative)",Xraw,64);
 report("QUAT associator (assoc≡0, control)",Xquat,54);
 report("OCT associator [h_t,h_{t+1},h_{t+2}]",Xoct,54);
 report("OCT associator + RAW (does it add?)",Xcat,118);
 printf("(Prior octonion methods on ABIDE ASD/TD: G2-Gram d=0.06 p=0.30; O-SSM 49.5%%. Honest replication expected near chance.)\n");
 return 0;
}

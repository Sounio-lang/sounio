#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
int status(double a,double b){double h=b-a;if(!std::isfinite(h)||!std::isfinite(a)||!std::isfinite(b))return 4;if(h==0||a+h!=b)return 2;double t2=a+h/5,t3=a+(.3)*h,t4=a+.8*h,t5=a+(8.0/9.0)*h;if(!(a<t2&&t2<t3&&t3<t4&&t4<t5&&t5<b))return 2;return 0;}
int main(int argc,char**argv){if(argc!=2)return 2;std::ifstream f(argv[1]);double current=0;bool found=false;for(std::string line;std::getline(f,line);){std::istringstream s(line);std::string tag,key;std::int64_t bits;if(s>>tag>>key>>bits&&tag=="DP54_VALUE"&&key=="time.tiny.time"){current=std::bit_cast<double>(std::uint64_t(bits));found=true;}}if(!found)return 3;double cap=1e-21,target=1e-20,candidate=current+cap;if(candidate-current>cap)candidate=std::nextafter(candidate,current);double tail=target-candidate;std::cout<<std::setprecision(17)<<"current="<<current<<" candidate="<<candidate<<" actual_h="<<candidate-current<<" cap="<<cap<<" target="<<target<<" tail="<<tail<<"\nfirst_status="<<status(current,candidate)<<" tail_status="<<status(candidate,target)<<" tail_sum_reaches_target="<<(candidate+tail==target)<<" tail_exceeds_cap="<<(tail>cap)<<'\n';if(status(current,candidate)!=0||status(candidate,target)!=2||!(tail>cap))return 4;std::cout<<"DP54_TINY_PLAN_DISCRIMINANT_PASS future_large_tail_is_not_final_sliver\n";return 0;}

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
using I=std::int64_t;
using Rows=std::map<std::string,std::string>;
static bool constraint(I kind,I a,I b){
    if(kind==0){return a==b;}
    if(kind<0||kind>6||a==-1){return false;}
    if(kind<=4&&b==-1){return false;}
    switch(kind){case 1:return a<b;case 2:return a<=b;case 3:return a>b;case 4:return a>=b;case 5:return a!=0;case 6:return a>0;default:return false;}
}
int main(int argc,char**argv){
    if(argc==5&&std::string(argv[1])=="mutate"){
        const std::string mode=argv[2];std::ifstream source(argv[3]);std::vector<std::string> lines;std::string line;
        while(std::getline(source,line)){lines.push_back(line);}
        if(lines.empty()){return 72;}
        if(mode=="duplicate"){lines.insert(lines.end()-1,lines.front());}
        else if(mode=="truncate"){lines.pop_back();}
        else if(mode=="wrong_value"){
            bool changed=false;for(auto& row:lines){if(row=="P|0|0"){row="P|0|1";changed=true;break;}}
            if(!changed){return 73;}
        }else{return 74;}
        std::ofstream output(argv[4]);for(const auto& row:lines){output<<row<<'\n';}return output?0:75;
    }
    if(argc!=4){return 64;}
    const std::string family=argv[1],variant=argv[2];
    if(variant!="before"&&variant!="after"){return 65;}
    Rows expected;
    if(family=="epistemic"){
        const std::vector<I> tags={0,1,2,3,4,5,6,7,-1,std::numeric_limits<I>::max(),std::numeric_limits<I>::min()};
        const std::vector<std::string> names={"derived","source","computed","literature","measured","input"};
        const double clocks[]={-1.0,.5,2.0};
        for(I kind:tags){
            const auto k=std::to_string(kind);
            if(variant=="after"||(kind>=0&&kind<6)){expected["N|"+k]=(kind>=0&&kind<6)?names.at(static_cast<std::size_t>(kind)):"unknown";}
            expected["P|"+k]=(kind==0||kind==1||kind==2||kind==3)?"0":"1";
            for(std::size_t j=0;j<3;++j){
                const bool expired=(kind<0||kind>3)||(kind==2&&clocks[j]>.5);
                expected["V|"+k+"|"+std::to_string(j)]=expired?"1":"0";
            }
        }
        expected["U|4"]="1";expected["S"]="0";
    }else if(family=="dependent"){
        const std::vector<std::pair<I,I>> values={{0,0},{-2,0},{2,-2},{3,3},{std::numeric_limits<I>::max(),0},{std::numeric_limits<I>::min(),0},{-1,0},{-1,-1}};
        for(I kind=0;kind<7;++kind){for(std::size_t j=0;j<values.size();++j){const auto[a,b]=values[j];expected["D|"+std::to_string(kind)+"|"+std::to_string(j)]=constraint(kind,a,b)?"1":"0";}}
        for(I kind:{I{-1},I{7},std::numeric_limits<I>::max(),std::numeric_limits<I>::min()}){expected["D|"+std::to_string(kind)+"|8"]="0";}
        for(I kind=0;kind<7;++kind){expected["A|"+std::to_string(kind)]=constraint(kind,5,5)?"1":"0";}
        expected["I|-1"]="0";expected["I|0"]="0";
    }else{return 66;}
    std::ifstream input(argv[3]);Rows observed;std::string line;bool marker=false;
    const auto end="TOTAL_TAGS_"+std::string(family=="epistemic"?"EPISTEMIC":"DEPENDENT")+"_END";
    while(std::getline(input,line)){
        if(line==end){if(marker){return 67;}marker=true;continue;}
        if(marker){return 68;}
        const auto pos=line.rfind('|');if(pos==std::string::npos||!observed.emplace(line.substr(0,pos),line.substr(pos+1)).second){return 69;}
    }
    if(!marker){std::cerr<<"missing complete marker\n";return 70;}
    if(observed!=expected){for(const auto&[k,v]:expected){auto it=observed.find(k);if(it==observed.end()||it->second!=v){std::cerr<<k<<" expected="<<v<<" observed="<<(it==observed.end()?"MISSING":it->second)<<'\n';}}if(observed.size()!=expected.size()){std::cerr<<"row count mismatch\n";}return 1;}
    std::cout<<"TOTAL_TAGS_CXX23_OK family="<<family<<" variant="<<variant<<" rows="<<observed.size()<<'\n';
}

#include <fstream>
#include <string>
#include <vector>
int main(int argc,char**argv){
    if(argc!=5){return 64;}
    const std::string family=argv[1],mode=argv[2];std::ifstream in(argv[3]);std::vector<std::string> lines;std::string row;
    while(std::getline(in,row)){lines.push_back(row);}if(lines.size()<2){return 65;}
    if(mode=="drop"){lines.erase(lines.begin());}
    else if(mode=="extra"){lines.insert(lines.end()-1,lines.front());}
    else if(mode=="wrong"){
        const std::string before=family=="graph"?"GRAPH_EDGES 4":"DP54_INVALID 0 3 0 0 3 0 0 0 0 0";
        const std::string after=family=="graph"?"GRAPH_EDGES 5":"DP54_INVALID 0 3 1 0 3 0 0 0 0 0";
        bool found=false;for(auto& line:lines){if(line==before){line=after;found=true;break;}}if(!found){return 66;}
    }else{return 67;}
    std::ofstream out(argv[4]);for(const auto& line:lines){out<<line<<'\n';}return out?0:68;
}

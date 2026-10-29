#include <array>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
static std::vector<std::string> model(){
    std::array<std::set<int>,4> adjacency;
    const std::array<std::pair<int,int>,4> edges={{{0,1},{0,2},{0,3},{1,2}}};
    for(const auto&[a,b]:edges){adjacency.at(a).insert(b);adjacency.at(b).insert(a);}
    std::vector<std::string> rows={"GRAPH_NODES "+std::to_string(adjacency.size()),"GRAPH_EDGES "+std::to_string(edges.size())};
    for(std::size_t i=0;i<adjacency.size();++i){rows.push_back("GRAPH_DEGREE "+std::to_string(i)+" "+std::to_string(adjacency[i].size()));}
    for(int i=0;i<4;++i){for(int j=0;j<4;++j){rows.push_back("GRAPH_HAS "+std::to_string(i)+" "+std::to_string(j)+" "+(adjacency.at(i).contains(j)?"1":"0"));}}
    rows.push_back("GRAPH_NEIGHBOR_COUNT 3 "+std::to_string(adjacency[3].size()));
    std::size_t k=0;for(int n:adjacency[3]){rows.push_back("GRAPH_NEIGHBOR 3 "+std::to_string(k++)+" "+std::to_string(n));}
    rows.push_back("GRAPH_METRICS_END");return rows;
}
int main(int argc,char**argv){
    if(argc!=2){return 64;}
    std::ifstream input(argv[1]);if(!input){return 65;}
    std::vector<std::string> observed;std::string line;while(std::getline(input,line)){observed.push_back(line);}
    const auto expected=model();
    if(observed!=expected){std::cerr<<"Graph actual metric/count/degree/adjacency/neighbor protocol mismatch\n";return 1;}
    std::cout<<"GRAPH_ADJACENCY_CPP23_OK nodes=4 edges=4 degree_and_all_pairs_and_neighbor=exact rows="<<observed.size()<<'\n';
}

#define main schema_parser_cli_main
#include "../wave4_loop_checkpoint_elision/nv2_trace_full_reference.cpp"
#undef main
#include <bitset>
#include <queue>
#include <map>
using Bits=std::bitset<512>;
struct Node{Row row;std::vector<int> next,pred;bool reachable=false;Bits out;};
int main(int argc,char**argv){try{
 if(argc!=2)return 64;
 const auto lines=load(argv[1]);Parser validated(lines);validated.verify(false,0);
 std::vector<Row> rows;for(const auto&s:lines)if(s.starts_with("NV2_IR_FULL_"))rows.push_back(row(s));
 std::size_t at=1;
 while(at<rows.size()&&rows[at].tag=="NV2_IR_FULL_FN"){
  const auto fn=rows[at++];const int fi=int(fn.v[0]),count=int(fn.v[7]),regs=int(fn.v[8]),pc=int(fn.v[10]);
  auto name=rows[at++];std::string label;for(std::size_t j=3;j<name.v.size();++j)label+=char(name.v[j]);
  ++at;const auto params=rows[at++];++at;
  std::vector<Node> nodes;std::map<I,int> labels;
  for(int i=0;i<count;++i){Node n;n.row=rows[at++];++at;if(n.row.v[3]==21)labels.emplace(n.row.v[11],i);nodes.push_back(n);}
  if(count==0||regs>512){std::cout<<"CFG_SKIP "<<fi<<" "<<label<<" count="<<count<<" regs="<<regs<<"\n";continue;}
  for(int i=0;i<count;++i){auto& n=nodes[i];const auto op=n.row.v[3];
   if(op==12||op==13||op==14){auto p=labels.find(n.row.v[11]);if(p==labels.end())bad("missing label");n.next.push_back(p->second);}
   if(op!=11&&op!=12&&i+1<count)n.next.push_back(i+1);
   for(int j:n.next)nodes[j].pred.push_back(i);
   n.out.set();
  }
  std::queue<int> q;q.push(0);nodes[0].reachable=true;while(!q.empty()){int i=q.front();q.pop();for(int j:nodes[i].next)if(!nodes[j].reachable){nodes[j].reachable=true;q.push(j);}}
  Bits entry;for(int i=0;i<pc;++i){if(params.v[std::size_t(i)+1]<0||params.v[std::size_t(i)+1]>=regs)bad("bad parameter");entry.set(std::size_t(params.v[std::size_t(i)+1]));}
  auto incoming=[&](int i){if(i==0)return entry;Bits value;value.set();bool found=false;for(int p:nodes[i].pred)if(nodes[p].reachable){value&=nodes[p].out;found=true;}return found?value:Bits{};};
  auto def=[](const Row&r){const auto op=r.v[3];return op==11||op==12||op==13||op==14||op==16||op==18||op==21||op==22?I(-1):r.v[4];};
  bool changed=true;int rounds=0;while(changed){changed=false;if(++rounds>count*regs+1)bad("dataflow did not converge");for(int i=0;i<count;++i)if(nodes[i].reachable){auto value=incoming(i);const auto d=def(nodes[i].row);if(d>=0){if(d>=regs)bad("definition bounds");value.set(std::size_t(d));}if(value!=nodes[i].out){nodes[i].out=value;changed=true;}}}
  int missing=0;
  for(int i=0;i<count;++i)if(nodes[i].reachable){
   const auto&r=nodes[i].row;const auto op=r.v[3];std::vector<I> uses;
   if(op==10)uses.assign(r.v.begin()+20,r.v.end());
   else if(op==8||op==16||op==17)uses={r.v[5],r.v[6]};
   else if(op==18)uses={r.v[5],r.v[6],I(bits(r.v[7],r.v[8]))};
   else if(op==2||op==3||op==7||op==9||op==11||op==13||op==14||op==15||op==19)uses={r.v[5]};
   const auto in=incoming(i);
   for(I reg:uses){if(reg<0||reg>=regs)bad("use bounds");if(!in.test(std::size_t(reg))){++missing;std::cout<<"CFG_UNDEFINED fi="<<fi<<" name="<<label<<" at="<<i<<" op="<<op<<" reg="<<reg<<"\n";}}
  }
  std::cout<<"CFG_MUSTDEF fi="<<fi<<" name="<<label<<" rounds="<<rounds<<" missing="<<missing<<"\n";
 }
 return 0;
}catch(const std::exception&e){std::cerr<<"CFG_AUDIT_REFUSED "<<e.what()<<"\n";return 1;}}

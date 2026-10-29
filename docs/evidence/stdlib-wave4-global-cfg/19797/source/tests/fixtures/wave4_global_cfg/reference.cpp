// ADR009 C++23. Global source-schema2 CFG must-def auditor.
// Strict parser is reused unchanged. Unsupported opcodes/ABI shapes refuse coverage.
#define main schema2_parser_cli_main
#include "../wave4_loop_checkpoint_elision/nv2_trace_full_reference.cpp"
#undef main
#include <filesystem>
#include <queue>
#include <set>
#include <map>
#include <bit>
struct Coverage:std::runtime_error{using std::runtime_error::runtime_error;};
[[noreturn]]static void unsupported(const std::string&s){throw Coverage(s);}
struct Sem{I def=-1;std::vector<I> use;};
static Sem semantics(const Row&r,I regs){
 const I op=r.v[3],dst=r.v[4],a=r.v[5],b=r.v[6];
 Sem s;
 for(I x:{dst,a,b})if(x< -1||x>=regs)bad("optional scalar register outside frame");
 switch(op){
 case 0:case 1:case 4:case 5:case 6:case 20:s.def=dst;break;
 case 2:case 3:case 7:case 9:case 15:case 19:s.def=dst;s.use={a};break;
 case 8:case 17:s.def=dst;s.use={a,b};break;
 case 10:
  s.def=dst;s.use.assign(r.v.begin()+20,r.v.end());break;
 case 11:case 13:case 14:s.use={a};break;
 case 12:case 21:case 22:break;
 case 16:s.use={a,b};break;
 case 18:s.use={a,b,std::bit_cast<I>(bits(r.v[7],r.v[8]))};break;
 default:unsupported("opcode="+std::to_string(op)+" has no certified operand/CFG semantics");
 }
 const bool needs_dst=(op<=9||op==15||op==17||op==19||op==20);
 if(needs_dst&&dst<0)bad("required destination absent");
 if(op!=10&&r.v[18]!=0)unsupported("counted arguments outside ordinary IrCall");
 if(s.def< -1||s.def>=regs)bad("destination outside register frame");
 for(I u:s.use)if(u<0||u>=regs)bad("semantic source outside register frame");
 return s;
}
struct Function{Row meta,params;std::string name;std::vector<Row> code;};
struct Module{std::vector<Function> functions;I instructions=0;};
static Module decode(const std::vector<std::string>&lines){
 Parser parser(lines);parser.verify(false,0);
 Module m;const auto&rows=parser.rows;std::size_t p=1;
 const I nf=rows.at(0).v.at(2);
 for(I fi=0;fi<nf;++fi){
  Function f;f.meta=rows.at(p++);
  const auto n=rows.at(p++);for(std::size_t j=3;j<n.v.size();++j)f.name+=char(n.v[j]);
  ++p;f.params=rows.at(p++);++p;
  for(I i=0;i<f.meta.v[7];++i){f.code.push_back(rows.at(p++));++p;++m.instructions;}
  m.functions.push_back(std::move(f));
 }
 if(p+1!=rows.size()||rows.at(p).tag!="NV2_IR_FULL_END"||I(m.functions.size())!=nf||m.instructions!=rows.at(0).v.at(3))bad("decoded census differs from strict parser");
 return m;
}
struct Bits{
 std::vector<std::uint64_t> v;
 explicit Bits(I regs,bool top=false):v(std::size_t((regs+63)/64),top?~UINT64_C(0):0){
  if(top&&regs%64&&!v.empty())v.back()=(UINT64_C(1)<<(regs%64))-1;
 }
 void set(I r){v.at(std::size_t(r/64))|=UINT64_C(1)<<(r%64);}
 bool test(I r)const{return(v.at(std::size_t(r/64))>>(r%64))&1;}
 void intersect(const Bits&b){for(std::size_t i=0;i<v.size();++i)v[i]&=b.v[i];}
 bool operator==(const Bits&)const=default;
};
struct Node{Sem sem;std::vector<I> next,pred;bool reachable=false;Bits out;explicit Node(I regs):out(regs,true){}};
struct Report{I functions=0,executed=0,empty=0,data=0,instructions=0,reachable=0,missing=0;};
static Report audit(const Module&m,std::ostream&out){
 Report result;result.functions=I(m.functions.size());result.instructions=m.instructions;
 for(const auto&f:m.functions){
  const auto&v=f.meta.v;const I fi=v[0],count=v[7],regs=v[8],pc=v[10];
  if(v[3]!=0||v[4]!=1||v[5]!=count||v[6]<count)bad("unsealed/invalid function descriptor");
  if(v[11]==99){
   ++result.data;out<<"GLOBAL_CFG_DESCRIPTOR fi="<<fi<<" kind=DATA name="<<f.name<<" instructions=0\n";continue;
  }
  if(count==0){
   ++result.empty;out<<"GLOBAL_CFG_DESCRIPTOR fi="<<fi<<" kind=EXTERN_EMPTY_DESCRIPTOR name="<<f.name<<" instructions=0\n";continue;
  }
  if(regs<0||regs>16384||count>16384)unsupported("register/instruction frame above official IR cap");
  if(v[21]!=0)unsupported("f128 parameter span ABI not certified");
  if(v[17]!=0&&v[17]!=1)unsupported("unknown sret metadata");
  ++result.executed;
  std::vector<Node> nodes;nodes.reserve(std::size_t(count));
  std::map<I,I> labels;
  for(I i=0;i<count;++i){
   nodes.emplace_back(regs);
   try{nodes.back().sem=semantics(f.code[std::size_t(i)],regs);}
   catch(const Coverage&e){unsupported("fi="+std::to_string(fi)+" at="+std::to_string(i)+" "+e.what());}
   if(f.code[std::size_t(i)].v[3]==21){
    const I label=f.code[std::size_t(i)].v[11];
    if(label<0||label>=v[9]||!labels.emplace(label,i).second)bad("duplicate/out-of-range label definition");
   }
  }
  auto edge=[&](I i,I target){auto&n=nodes[std::size_t(i)];if(std::find(n.next.begin(),n.next.end(),target)==n.next.end()){n.next.push_back(target);nodes[std::size_t(target)].pred.push_back(i);}};
  for(I i=0;i<count;++i){
   const auto&r=f.code[std::size_t(i)];const I op=r.v[3];
   if(op==12||op==13||op==14){
    const auto found=labels.find(r.v[11]);if(found==labels.end())bad("jump to absent label");
    edge(i,found->second);
   }
   if(op!=11&&op!=12&&i+1<count)edge(i,i+1);
  }
  std::queue<I> work;work.push(0);nodes[0].reachable=true;
  while(!work.empty()){const I i=work.front();work.pop();for(I j:nodes[std::size_t(i)].next)if(!nodes[std::size_t(j)].reachable){nodes[std::size_t(j)].reachable=true;work.push(j);}}
  if(nodes.back().reachable&&f.code.back().v[3]!=11&&f.code.back().v[3]!=12)unsupported("reachable function fallthrough has no explicit return/jump semantics");
  Bits entry(regs);std::set<I> actual_params;
  for(I j=0;j<pc;++j){const I r=f.params.v[std::size_t(j+1)];if(r<0||r>=regs||!actual_params.insert(r).second)bad("invalid/duplicate actual parameter register");entry.set(r);}
  if(v[17]==1){if(v[18]<0||v[18]>=regs)bad("sret entry register absent");entry.set(v[18]);}
  auto incoming=[&](I i){
   if(i==0)return entry; // implicit program-entry edge prevents loopback-created entry defs
   Bits value(regs,true);bool any=false;
   for(I p:nodes[std::size_t(i)].pred)if(nodes[std::size_t(p)].reachable){value.intersect(nodes[std::size_t(p)].out);any=true;}
   if(!any)bad("reachable instruction without reachable predecessor");
   return value;
  };
  bool changed=true;I rounds=0;
  while(changed){
   changed=false;if(++rounds>count*(regs+1)+1)bad("must-definition fixed point did not converge");
   for(I i=0;i<count;++i)if(nodes[std::size_t(i)].reachable){
    Bits value=incoming(i);const I d=nodes[std::size_t(i)].sem.def;if(d>=0)value.set(d);
    if(!(value==nodes[std::size_t(i)].out)){nodes[std::size_t(i)].out=std::move(value);changed=true;}
   }
  }
  I reachable=0,missing=0;
  for(I i=0;i<count;++i)if(nodes[std::size_t(i)].reachable){
   ++reachable;const Bits in=incoming(i);
   for(I r:nodes[std::size_t(i)].sem.use)if(!in.test(r)){
    ++missing;out<<"GLOBAL_CFG_UNDEFINED fi="<<fi<<" at="<<i<<" op="<<f.code[std::size_t(i)].v[3]<<" reg="<<r<<" name="<<f.name<<"\n";
   }
  }
  result.reachable+=reachable;result.missing+=missing;
  out<<"GLOBAL_CFG_FUNCTION fi="<<fi<<" instructions="<<count<<" reachable="<<reachable<<" parameters="<<pc<<" rounds="<<rounds<<" missing="<<missing<<" name="<<f.name<<"\n";
 }
 if(result.functions!=result.executed+result.empty+result.data)bad("global function coverage mismatch");
 if(result.executed==0)unsupported("empty executable corpus");
 out<<"GLOBAL_CFG_SUMMARY functions="<<result.functions<<" executable="<<result.executed<<" extern_empty="<<result.empty<<" data="<<result.data<<" instructions="<<result.instructions<<" reachable="<<result.reachable<<" total_missing="<<result.missing<<" coverage=complete\n";
 return result;
}
// Synthetic full schema2 producers for analysis controls only. Never used as compiler evidence.
struct Ins{
 I op,dst,a,b,label,imm;std::vector<I>args;
 Ins(I op_,I dst_=-1,I a_=-1,I b_=-1,I label_=-1,I imm_=0,std::vector<I>args_={}):
 op(op_),dst(dst_),a(a_),b(b_),label(label_),imm(imm_),args(std::move(args_)){}
};
struct Spec{std::string name;I regs=0;std::vector<I>params;std::vector<Ins>code;bool data=false;};
static std::vector<std::string> fixture(const std::vector<Spec>&specs){
 std::vector<std::string> lines;I total=0,args=0,names=0;
 for(const auto&s:specs){total+=I(s.code.size());names+=I(s.name.size());for(const auto&i:s.code)args+=I(i.args.size());}
 lines.push_back(render(Row{"NV2_IR_FULL_BEGIN",{2,0,I(specs.size()),total,args,names,1,args,total+1,0}}));
 I slot=0,argbase=0;
 for(std::size_t fi=0;fi<specs.size();++fi){
  const auto&s=specs[fi];I labels=0;for(const auto&i:s.code)if(i.op==21)labels=std::max(labels,i.label+1);
  Row f{"NV2_IR_FULL_FN",{I(fi),I(fi+1),1,0,1,I(s.code.size()),std::max(I(16),I(s.code.size())),I(s.code.size()),s.regs,labels,I(s.params.size()),s.data?99:0,0,-1,-1,-1,-1,0,-1,0,0,0,0}};
  lines.push_back(render(f));Row name{"NV2_IR_FULL_FN_NAME",{I(fi),-1,I(s.name.size())}};for(unsigned char c:s.name)name.v.push_back(c);lines.push_back(render(name));
  lines.push_back(render(Row{"NV2_IR_FULL_RETURN_NAME",{I(fi),-1,0}}));
  Row params{"NV2_IR_FULL_PARAMS",{I(fi)}};for(I j=0;j<64;++j)params.v.push_back(j<I(s.params.size())?s.params[std::size_t(j)]:-1);lines.push_back(render(params));
  lines.push_back(render(Row{"NV2_IR_FULL_FLOAT_BITS",{I(fi),1,0}}));
  I ii=0;
  for(const auto&i:s.code){
   const auto u=std::bit_cast<std::uint64_t>(i.imm);
   Row ins{"NV2_IR_FULL_INSTR",{I(fi),ii,slot,i.op,i.dst,i.a,i.b,I(u>>32),I(u&UINT64_C(0xffffffff)),0,0,i.label,-1,-1,0,0,0,0,I(i.args.size()),argbase}};
   ins.v.insert(ins.v.end(),i.args.begin(),i.args.end());lines.push_back(render(ins));
   lines.push_back(render(Row{"NV2_IR_FULL_INSTR_NAME",{I(fi),ii,0}}));
   ++ii;++slot;argbase+=I(i.args.size());
  }
 }
 lines.push_back(render(Row{"NV2_IR_FULL_END",{2,0,I(specs.size()),total,args,names}}));return lines;
}
static int execute(const std::vector<std::string>&lines,std::ostream&out){
 try{const auto r=audit(decode(lines),out);return r.missing?1:0;}
 catch(const Coverage&e){out<<"GLOBAL_CFG_COVERAGE_REFUSED "<<e.what()<<"\n";return 2;}
 catch(const std::exception&e){out<<"GLOBAL_CFG_STRUCTURE_REFUSED "<<e.what()<<"\n";return 3;}
}
static void controls(const std::string&dir){
 std::filesystem::create_directory(dir);
 const Spec linear{"nonstandard_param",5,{3},{{7,4,3},{11,-1,4}}};
 const Spec branch{"branch",3,{0},{{14,-1,0,-1,0},{0,1},{12,-1,-1,-1,1},{21,-1,-1,-1,0},{0,1},{21,-1,-1,-1,1},{11,-1,1}}};
 const Spec loop{"loop",3,{0},{{0,1},{21,-1,-1,-1,0},{14,-1,0,-1,1},{8,1,1,0},{12,-1,-1,-1,0},{21,-1,-1,-1,1},{11,-1,1}}};
 const Spec empty{"extern_empty",0,{},{}},data{"DATA",0,{},{},true};
 I passed=0;
 auto test=[&](const std::string&name,std::vector<std::string> lines,int expected){
  const auto p=dir+"/"+name+".capture";write(p.c_str(),lines);std::ostringstream log;const int got=execute(lines,log);write((dir+"/"+name+".audit").c_str(),{log.str()});
  if(got!=expected)bad("control "+name+" expected="+std::to_string(expected)+" got="+std::to_string(got));
  std::cout<<"GLOBAL_CFG_CONTROL "<<name<<" status="<<got<<"\n";++passed;
 };
 test("correct",fixture({linear,empty,data}),0);
 test("correct-branch",fixture({branch}),0);
 test("correct-loop",fixture({loop}),0);
 auto missing=branch;missing.code[4]={22};test("missing-branch",fixture({missing}),1);
 auto loop_missing=loop;loop_missing.code[0]={22};test("missing-loop-entry",fixture({loop_missing}),1);
 auto ordering=linear;ordering.code={{7,4,1},{0,1},{11,-1,4}};test("same-block-order",fixture({ordering}),1);
 auto badparam=linear;badparam.params={0};test("wrong-actual-param",fixture({badparam}),1);
 auto high=linear;high.regs=1025;high.params={1024};high.code={{11,-1,1024}};test("above-old-512-cap",fixture({high}),0);
 auto unknown=linear;unknown.code[0].op=250;test("unknown-op",fixture({unknown}),2);
 auto phi=linear;phi.code[0].op=23;test("phi-coverage-refusal",fixture({phi}),2);
 auto absent=branch;absent.code[0].label=9;test("absent-label",fixture({absent}),3);
 auto duplicate=branch;duplicate.code[5].label=0;test("duplicate-label",fixture({duplicate}),3);
 auto noexec=fixture({empty,data});test("empty-executable-corpus",noexec,2);
 auto drop=fixture({linear,empty});auto it=std::find_if(drop.begin(),drop.end(),[](const auto&s){return s.starts_with("NV2_IR_FULL_FN 1 ");});drop.erase(it,drop.end()-1);test("drop-function",drop,3);
 auto bad_data=data;bad_data.regs=1;test("DATA-shape",fixture({linear,bad_data}),3);
 auto bad_empty=empty;bad_empty.regs=1;test("extern-empty-shape",fixture({linear,bad_empty}),3);
 auto unreachable=linear;unreachable.code={{11,-1,3},{7,4,1}};test("unreachable-use-not-counted",fixture({unreachable}),0);
 auto indirect=linear;indirect.code[0].op=50;test("unsupported-indirect",fixture({indirect}),2);
 auto fall=linear;fall.code={{7,4,3}};test("fallthrough-coverage",fixture({fall}),2);
 auto call=linear;call.code={{10,4,-1,-1,-1,0,{3}},{11,-1,4}};test("call-args",fixture({call}),0);
 call.code[0].args={1};test("call-arg-missing",fixture({call}),1);
 std::cout<<"GLOBAL_CFG_CONTROLS_OK count="<<passed<<"\n";
}
int main(int argc,char**argv){
 try{
  if(argc==3&&std::string_view(argv[1])=="--audit")return execute(load(argv[2]),std::cout);
  if(argc==3&&std::string_view(argv[1])=="--self-controls"){controls(argv[2]);return 0;}
  return 64;
 }catch(const std::exception&e){std::cerr<<"GLOBAL_CFG_COMPONENT_ERROR "<<e.what()<<"\n";return 3;}
}

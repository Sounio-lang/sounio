// C++23 complete-manifest collector. Metadata acceptance, not a Unicode implementation.
#define SOUNIO_CHUNK_RUNNER_NO_MAIN
#include "run_chunks.cpp"
#include <map>
#include <cctype>
#include <set>
#include <limits>
void add(N&sum,N value){if(value>std::numeric_limits<N>::max()-sum)fail("count overflow");sum+=value;}
std::vector<std::string> fields(const std::string&s){std::istringstream in(s);std::vector<std::string> out;std::string part;while(std::getline(in,part,'\t'))out.push_back(part);return out;}
std::map<std::string,N> status_file(const fs::path&dir){std::ifstream f(dir/"STATUS");std::string line;if(!std::getline(f,line)||line!="UNICODE_CHUNK_PART_OK")fail("part status marker");std::map<std::string,N> out;while(std::getline(f,line)){auto eq=line.find('=');if(eq==line.npos||!out.emplace(line.substr(0,eq),number(line.substr(eq+1))).second)fail("status field");}if(out.size()!=4||!out.contains("part")||!out.contains("parts")||!out.contains("files")||!out.contains("rows"))fail("status keys");return out;}
int main(int argc,char**argv){try{
 std::string coherent_kind;
 if(argc>1&&std::string(argv[1])=="--coherent-kind") {
  constexpr int family=SOUNIO_CHUNK_KIND_MIN==5?2:SOUNIO_CHUNK_KIND_MIN==6?3:1;
  if(argc<7||std::string(argv[2])!=std::to_string(family))return 64;
  coherent_kind=argv[2];argc-=2;argv+=2;
 }
 if(argc<5){std::cerr<<"usage: collect_chunks FIXTURE_DIR EXPECTED_ELF_SHA256 NEW_SUMMARY_DIR PART_RUN_DIR...\n";return 64;}
 std::string expected_sha=argv[2];if(expected_sha.size()!=64||!std::all_of(expected_sha.begin(),expected_sha.end(),[](unsigned char c){return std::isxdigit(c);}))fail("expected ELF hash");
 fs::path root=fs::canonical(argv[1]),summary=fs::absolute(argv[3]);auto all=manifest(root);std::map<std::string,std::size_t> index;for(std::size_t i=0;i<all.size();++i)if(!index.emplace(all[i].name,i).second)fail("duplicate manifest fixture");
 std::set<std::string> seen;std::set<N> parts_seen;N common_parts=0,total_rows=0;std::map<N,std::pair<N,N>> per_kind;
 for(int arg=4;arg<argc;++arg){auto dir=fs::canonical(argv[arg]);std::ifstream identity(dir.parent_path()/"EXECUTABLE.sha256");std::string sha;identity>>sha;if(!identity||sha!=expected_sha)fail("part ELF identity");
  if(!coherent_kind.empty()) {
   std::ifstream invocation(dir/"ELF_INVOCATION.tsv",std::ios::binary);
   const std::string actual((std::istreambuf_iterator<char>(invocation)),{});
   if(!invocation||actual!="argc\t1\nkind\t"+coherent_kind+"\n")fail("ELF invocation kind/argc binding");
  }
  auto status=status_file(dir);N part=status.at("part"),parts=status.at("parts");if(parts==0||parts>1024||part>=parts||!parts_seen.insert(part).second)fail("invalid/duplicate partition");if(!common_parts)common_parts=parts;else if(parts!=common_parts)fail("partition topology mismatch");
  std::ifstream f(dir/"RESULTS.tsv");std::string line;if(!std::getline(f,line)||line!="fixture\tkind\texpected_rows\texit_status\tmarker\tactual_rows")fail("results header");N files=0,rows=0;
  while(std::getline(f,line)){auto cols=fields(line);if(cols.size()!=6)fail("results fields");auto found=index.find(cols[0]);if(found==index.end()||!seen.insert(cols[0]).second||found->second%parts!=part)fail("unknown/duplicate/mispartitioned fixture");const auto&expected=all[found->second];if(number(cols[1])!=expected.kind||number(cols[2])!=expected.rows||number(cols[3])!=0||cols[4]!="1"||number(cols[5])!=expected.rows)fail("failed/inconsistent fixture row");add(files,1);add(rows,expected.rows);add(total_rows,expected.rows);add(per_kind[expected.kind].first,1);add(per_kind[expected.kind].second,expected.rows);}
  if(!files||files!=status.at("files")||rows!=status.at("rows"))fail("empty/incomplete part counts");
 }
 if(parts_seen.size()!=common_parts||seen.size()!=all.size())fail("incomplete corpus/partition coverage");
 for(N part=0;part<common_parts;++part)if(!parts_seen.contains(part))fail("missing partition");
 if(!fs::create_directory(summary))fail("summary directory must be new");std::ofstream scope(summary/"FIXTURE_SCOPE.txt");std::ifstream source_scope(root/"SCOPE.txt");scope<<source_scope.rdbuf();if(!scope)fail("fixture scope output");
 std::ofstream report(summary/"SUMMARY.tsv");report<<"kind\tfiles\trows\n";for(const auto&[kind,counts]:per_kind)report<<kind<<'\t'<<counts.first<<'\t'<<counts.second<<'\n';if(!report)fail("summary output");std::ofstream complete(summary/"STATUS");complete<<"UNICODE_COMPLETE_MANIFEST_OK\nfiles="<<seen.size()<<"\nrows="<<total_rows<<"\npartitions="<<common_parts<<"\nelf_sha256="<<expected_sha<<"\n";complete.flush();if(!complete)fail("completion status write");
 std::cout<<"UNICODE_COMPLETE_MANIFEST_OK files="<<seen.size()<<" rows="<<total_rows<<" partitions="<<common_parts<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"UNICODE_COLLECTOR_ERROR "<<e.what()<<"\n";return 1;}}

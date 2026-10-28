// BIDI-only orchestration. Production resolution and expected corpus are unchanged.
// Shared exec accepts explicit argv; this tool never substitutes a wrapper for ELF.
#define SOUNIO_CHUNK_RUNNER_NO_MAIN
#include "run_chunks.cpp"
#include <map>
#include <cctype>
#include <set>
#include <limits>
struct BidiChunk {std::string file;N cases,bytes;};
std::vector<BidiChunk> bidi_manifest(const fs::path& root) {
 std::ifstream in(root/"CHUNKS.tsv");if(!in)fail("BIDI manifest missing");
 std::string line;std::vector<BidiChunk> result;std::set<std::string> seen;N total=0;
 const std::regex safe("^[0-9]{6}\\.bin$");
 while(std::getline(in,line)) {
  if(std::count(line.begin(),line.end(),'\t')!=2)fail("BIDI manifest fields");
  std::istringstream row(line);std::array<std::string,3> fields;
  for(auto& field:fields)if(!std::getline(row,field,'\t'))fail("BIDI manifest fields");
  BidiChunk chunk{fields[0],number(fields[1]),number(fields[2])};
  if(!std::regex_match(chunk.file,safe)||!seen.insert(chunk.file).second||
     chunk.cases==0||chunk.cases>4096||chunk.bytes<16||chunk.bytes>1048576)fail("BIDI manifest value");
  if(fs::file_size(root/chunk.file)!=chunk.bytes)fail("BIDI chunk bytes");
  std::ifstream file(root/chunk.file,std::ios::binary);std::array<unsigned char,24> header{};
  file.read(reinterpret_cast<char*>(header.data()),16);
  if(!file||word(header,0)!=0x44494253||word(header,4)!=0x00383149||
     word(header,8)!=1||word(header,12)!=chunk.cases)fail("BIDI header");
  if(chunk.cases>861948-total)fail("BIDI case overflow");
  total+=chunk.cases;result.push_back(std::move(chunk));
 }
 if(result.size()!=211||total!=861948)fail("full Unicode18 BIDI inventory required");
 return result;
}
std::pair<bool,N> bidi_observed(const fs::path& file) {
 std::ifstream in(file);if(!in)fail("BIDI stdout missing");
 std::string line;N cases=0;int markers=0,counts=0;
 while(std::getline(in,line)) {
  if(line=="BIDI_NATIVE_CONFORMANCE_OK")++markers;
  if(line.starts_with("BIDI_CASES_OK ")) {cases=number(line.substr(14));++counts;}
 }
 return {markers==1&&counts==1,cases};
}
std::map<std::string,N> bidi_status(const fs::path& dir) {
 std::ifstream in(dir/"STATUS");std::string line;
 if(!std::getline(in,line)||line!="BIDI_COHERENT_PART_OK")fail("BIDI part status");
 std::map<std::string,N> out;
 while(std::getline(in,line)) {
  const auto eq=line.find('=');
  if(eq==line.npos||!out.emplace(line.substr(0,eq),number(line.substr(eq+1))).second)fail("BIDI status field");
 }
 for(const auto* name:{"part","parts","files","cases","kind","argc"})if(!out.contains(name))fail("BIDI missing status key");
 if(out.size()!=6||out.at("kind")!=4||out.at("argc")!=1)fail("BIDI argv binding");
 return out;
}
void sum(N& dest,N value){if(value>std::numeric_limits<N>::max()-dest)fail("count overflow");dest+=value;}
int bidi_run(int argc,char**argv) {
 if(argc!=8||std::string(argv[6])!="--coherent-kind"||std::string(argv[7])!="4")return 64;
 const fs::path elf=fs::canonical(argv[1]),root=fs::canonical(argv[2]),out=fs::absolute(argv[3]);
 const N part=number(argv[4]),parts=number(argv[5]);
 if(parts==0||parts>211||part>=parts)fail("BIDI partition");
 const auto all=bidi_manifest(root);std::vector<BidiChunk> selected;
 for(std::size_t i=0;i<all.size();++i)if(i%parts==part)selected.push_back(all[i]);
 if(selected.empty())fail("BIDI empty selection");
 if(!fs::create_directory(out))fail("BIDI fresh output required");
 std::ofstream invocation(out/"ELF_INVOCATION.tsv");invocation<<"argc\t1\nkind\t4\n";invocation.close();
 if(!invocation)fail("BIDI invocation output");
 std::ofstream rows(out/"RESULTS.tsv");rows<<"fixture\tcases\tbytes\texit_status\tmarker\tactual_cases\tkind\targc\n";
 N completed=0,total=0;
 for(const auto& c:selected) {
  const auto cwd=out/fs::path(c.file).stem(),destination=cwd/"data/current.bin";
  fs::create_directories(destination.parent_path());fs::copy_file(root/c.file,destination);
  const int status=execute(elf,cwd,"4");const auto [marker,count]=bidi_observed(cwd/"native.stdout");
  rows<<c.file<<'\t'<<c.cases<<'\t'<<c.bytes<<'\t'<<status<<'\t'<<marker<<'\t'<<count<<"\t4\t1\n";
  rows.flush();if(!rows)fail("BIDI results output");
  if(status!=0||!marker||count!=c.cases)fail("BIDI native chunk failed "+c.file);
  ++completed;sum(total,count);
 }
 std::ofstream status(out/"STATUS");
 status<<"BIDI_COHERENT_PART_OK\npart="<<part<<"\nparts="<<parts<<"\nfiles="<<completed<<"\ncases="<<total<<"\nkind=4\nargc=1\n";
 status.close();if(!status)fail("BIDI status output");
 std::cout<<"BIDI_COHERENT_PART_OK part="<<part<<" parts="<<parts<<" files="<<completed<<" cases="<<total<<'\n';
 return 0;
}
int bidi_collect(int argc,char**argv) {
 if(argc<8||std::string(argv[2])!="--coherent-kind"||std::string(argv[3])!="4")return 64;
 const fs::path root=fs::canonical(argv[4]),out=fs::absolute(argv[6]);
 const std::string expected_sha=argv[5];
 if(expected_sha.size()!=64||!std::all_of(expected_sha.begin(),expected_sha.end(),[](unsigned char c){return std::isxdigit(c);}))fail("BIDI ELF digest");
 const auto all=bidi_manifest(root);std::map<std::string,std::size_t> index;
 for(std::size_t i=0;i<all.size();++i)index.emplace(all[i].file,i);
 std::set<std::string> seen;std::set<N> seen_parts;N common_parts=0,total=0;
 for(int arg=7;arg<argc;++arg) {
  const fs::path dir=fs::canonical(argv[arg]);
  std::ifstream identity(dir.parent_path()/"EXECUTABLE.sha256");std::string sha;identity>>sha;
  if(!identity||sha!=expected_sha)fail("BIDI ELF identity");
  std::ifstream invocation(dir/"ELF_INVOCATION.tsv",std::ios::binary);
  const std::string binding((std::istreambuf_iterator<char>(invocation)),{});
  if(!invocation||binding!="argc\t1\nkind\t4\n")fail("BIDI invocation binding");
  const auto status=bidi_status(dir);const N part=status.at("part"),parts=status.at("parts");
  if(parts==0||parts>211||part>=parts||!seen_parts.insert(part).second)fail("BIDI duplicate/invalid part");
  if(!common_parts)common_parts=parts;else if(parts!=common_parts)fail("BIDI partition topology");
  std::ifstream input(dir/"RESULTS.tsv");std::string line;
  if(!std::getline(input,line)||line!="fixture\tcases\tbytes\texit_status\tmarker\tactual_cases\tkind\targc")fail("BIDI result header");
  N files=0,cases=0;
  while(std::getline(input,line)) {
   if(std::count(line.begin(),line.end(),'\t')!=7)fail("BIDI result fields");
   std::istringstream row(line);std::array<std::string,8> fields;
   for(auto& field:fields)if(!std::getline(row,field,'\t'))fail("BIDI result fields");
   const auto found=index.find(fields[0]);
   if(found==index.end()||!seen.insert(fields[0]).second||found->second%parts!=part)fail("BIDI fixture coverage");
   const auto& expected=all[found->second];
   if(number(fields[1])!=expected.cases||number(fields[2])!=expected.bytes||fields[3]!="0"||
      fields[4]!="1"||number(fields[5])!=expected.cases||fields[6]!="4"||fields[7]!="1")fail("BIDI result mismatch");
   ++files;sum(cases,expected.cases);sum(total,expected.cases);
  }
  if(files==0||files!=status.at("files")||cases!=status.at("cases"))fail("BIDI part counts");
 }
 if(seen_parts.size()!=common_parts||seen.size()!=211||total!=861948)fail("BIDI incomplete full corpus");
 for(N part=0;part<common_parts;++part)if(!seen_parts.contains(part))fail("BIDI missing part");
 if(!fs::create_directory(out))fail("BIDI fresh summary required");
 std::ofstream report(out/"STATUS");
 report<<"BIDI_COHERENT_FULL_NATIVE_OK\nfiles=211\ncases=861948\nkind=4\nargc=1\nelf_sha256="<<expected_sha<<'\n';
 report.close();if(!report)fail("BIDI summary output");
 std::cout<<"BIDI_COHERENT_FULL_NATIVE_OK files=211 cases=861948 kind=4 argc=1\n";
 return 0;
}
int main(int argc,char**argv) {
 try {return argc>1&&std::string(argv[1])=="--collect"?bidi_collect(argc,argv):bidi_run(argc,argv);}
 catch(const std::exception& e){std::cerr<<"BIDI_COHERENT_ERROR "<<e.what()<<'\n';return 1;}
}

// C++23 fixture compiler. Expected normalization/segmentation is from pinned UCD;
// casing expectations are produced only by the isolated ICU79.1 test oracle.
#define main sounio_unicode_reference_main
#include "wave3_unicode_reference.cpp"
#undef main
#include <iomanip>
#include <set>
using FixtureBytes=std::vector<std::uint8_t>;
void fixture_put(FixtureBytes& b,U x){for(U i=0;i<4;++i)b.push_back(static_cast<std::uint8_t>((x>>(8*i))&255));}
void fixture_seq(FixtureBytes& b,const Seq& seq){if(seq.size()>std::numeric_limits<U>::max())fail("fixture sequence too large");fixture_put(b,static_cast<U>(seq.size()));for(U cp:seq)fixture_put(b,cp);}
void fixture_word(std::ostream& out,U x){std::array<char,4> b{};for(U i=0;i<4;++i)b[i]=static_cast<char>((x>>(8*i))&255);out.write(b.data(),b.size());}
struct FixtureWriter{
 std::filesystem::path directory;std::string prefix;U kind,serial=0,rows=0,payload=0;std::uint64_t total=0;std::ofstream out;std::ostream& manifest;std::string filename;
 FixtureWriter(std::filesystem::path d,std::string p,U k,std::ostream& m):directory(std::move(d)),prefix(std::move(p)),kind(k),manifest(m){}
 void finish(){
  if(!out.is_open())return;
  out.seekp(12);fixture_word(out,rows);fixture_word(out,payload);out.close();if(!out)fail("fixture close");
  manifest<<filename<<'\t'<<kind<<'\t'<<rows<<'\t'<<payload<<'\n';++serial;rows=0;payload=0;
 }
 void add(const FixtureBytes& row){
  if(row.size()>1048576)fail("fixture row exceeds1MiB");
  if(out.is_open()&&(rows==512||std::uint64_t{payload}+row.size()>1048576))finish();
  if(!out.is_open()){
   std::ostringstream name;name<<prefix<<'-'<<std::setw(6)<<std::setfill('0')<<serial<<".bin";filename=name.str();out.open(directory/filename,std::ios::binary);if(!out)fail("fixture output");
   out.write("SOUNUTF1",8);fixture_word(out,kind);fixture_word(out,0);fixture_word(out,0);fixture_word(out,0x00120000);
  }
  out.write(reinterpret_cast<const char*>(row.data()),static_cast<std::streamsize>(row.size()));if(!out)fail("fixture write");++rows;++total;payload+=static_cast<U>(row.size());
 }
};
U fixture_utf8_width(U cp){if(cp>0x10FFFF||(cp>=0xD800&&cp<=0xDFFF))fail("non-scalar conformance input");return cp<0x80?1:cp<0x800?2:cp<0x10000?3:4;}
void fixture_normalization(const std::filesystem::path& root,FixtureWriter& writer){
 std::ifstream input(root/"NormalizationTest.txt");if(!input)fail("NormalizationTest input");std::string line;
 while(std::getline(input,line)){
  line=trim(line.substr(0,line.find('#')));if(line.empty()||line.front()=='@')continue;auto fields=split(line,';');if(fields.size()<5)fail("NormalizationTest fields");
  FixtureBytes row;for(U i=0;i<5;++i)fixture_seq(row,parse(fields[i]));writer.add(row);
 }
}
void fixture_segmentation(const std::filesystem::path& root,FixtureWriter& writer){
 const std::array<const char*,3> names={"Grapheme","Word","Sentence"};
 for(U mode=0;mode<3;++mode){
  std::ifstream input(root/"auxiliary"/(std::string(names[mode])+"BreakTest.txt"));if(!input)fail("break test input");std::string line;
  while(std::getline(input,line)){
   line=trim(line.substr(0,line.find('#')));if(line.empty())continue;Seq seq,expected;std::istringstream tokens(line);std::string token;U bytes=0;
   while(tokens>>token){if(token=="\xC3\xB7")expected.push_back(bytes);else if(token!="\xC3\x97"){U cp=static_cast<U>(std::stoul(token,nullptr,16));seq.push_back(cp);bytes+=fixture_utf8_width(cp);}}
   FixtureBytes row;fixture_put(row,mode);fixture_seq(row,seq);fixture_seq(row,expected);writer.add(row);
  }
 }
}
Seq fixture_fold(const Seq& input,bool turkic){
 auto in=utf16(input);std::array<UChar,512> out{};UErrorCode error=U_ZERO_ERROR;
 auto n=u_strFoldCase(out.data(),out.size(),in.data(),static_cast<int32_t>(in.size()),turkic?U_FOLD_CASE_EXCLUDE_SPECIAL_I:U_FOLD_CASE_DEFAULT,&error);
 if(U_FAILURE(error)||n<0||static_cast<std::size_t>(n)>out.size())fail("fixture fold ICU error");
 return scalars(std::span<const UChar>(out.data(),static_cast<std::size_t>(n)));
}
void fixture_cases(const Seq& input,FixtureWriter& writer){
 for(U locale=0;locale<4;++locale)for(U op=0;op<2;++op){
  FixtureBytes row;fixture_put(row,locale);fixture_put(row,op);fixture_seq(row,input);fixture_seq(row,icu_case(input,op==1,locale));writer.add(row);
 }
 for(U op=2;op<4;++op){FixtureBytes row;fixture_put(row,0);fixture_put(row,op);fixture_seq(row,input);fixture_seq(row,fixture_fold(input,op==3));writer.add(row);}
}
void fixture_casing(const std::filesystem::path& root,bool exhaustive,FixtureWriter& writer){
 if(exhaustive){for(U cp=0;cp<0x110000;++cp)if(cp<0xD800||cp>0xDFFF)fixture_cases(Seq{cp},writer);}
 else{
  std::set<U> selected={0,1,0x7F,0x80,0x7FF,0x800,0xD7FF,0xE000,0xFFFF,0x10000,0x10FFFF};
  auto collect=[&](const std::filesystem::path& path,char sep,bool unicode){
   std::ifstream input(path);if(!input)fail("casing fixture input");std::string line;
   while(std::getline(input,line)){line=trim(line.substr(0,line.find('#')));if(line.empty())continue;auto fields=split(line,sep);if(!unicode||(!fields[12].empty()||!fields[13].empty()||!fields[14].empty()))selected.insert(static_cast<U>(std::stoul(fields[0],nullptr,16)));}
  };
  collect(root/"UnicodeData.txt",';',true);collect(root/"CaseFolding.txt",';',false);collect(root/"SpecialCasing.txt",';',false);
  for(U cp:selected)fixture_cases(Seq{cp},writer);
 }
 const Seq alphabet={0,0x20,0x21,0x41,0x61,0x49,0x69,0x4A,0xDF,0x130,0x12E,0x307,0x300,0x323,0x345,0x2B0,0x3A3};
 const Seq targets={0x49,0x4A,0x12E,0xCC,0xCD,0x128,0x130,0x69,0x307,0x3A3,0xDF,0xFB03};
 for(U target:targets)for(U before:alphabet)for(U after:alphabet){
  fixture_cases(Seq{before,target,after},writer);
  fixture_cases(Seq{before,0x323,target,0x323,after},writer);
  fixture_cases(Seq{before,0x300,target,0x300,after},writer);
 }
}

void fixture_smoke(const std::filesystem::path& out){
 auto dir=out/"smoke";std::filesystem::create_directory(dir);std::ofstream manifest(dir/"MANIFEST.tsv");manifest<<"filename\tkind\trows\tpayload_bytes\n";
 FixtureWriter norm(dir,"norm",1,manifest),segment(dir,"segment",2,manifest),casing(dir,"case",3,manifest);
 UErrorCode error=U_ZERO_ERROR;std::array<const UNormalizer2*,4> forms={unorm2_getNFCInstance(&error),unorm2_getNFDInstance(&error),unorm2_getNFKCInstance(&error),unorm2_getNFKDInstance(&error)};if(U_FAILURE(error))fail("smoke normalizer");
 const std::vector<Seq> normalization_inputs={{0},{0xE9},{0x65,0x301},{0xAC01},{0xFB03},{0x41,0x323,0x30A}};
 for(const auto& input:normalization_inputs){FixtureBytes row;fixture_seq(row,input);for(auto form:forms)fixture_seq(row,icu_normalize(form,input));norm.add(row);}norm.finish();
 const std::vector<Seq> boundary_inputs={{0x94D,0x915},{0x1F469,0x200D,0x1F52C},{0x1F1E6,0x1F1E7,0x1F1E8},{0x61,0,0x62},{0x61,0x2E,0x42,0x20,0x63}};
 for(U mode=0;mode<3;++mode)for(const auto& input:boundary_inputs){
  FixtureBytes row;fixture_put(row,mode);fixture_seq(row,input);Seq prefix={0};for(U cp:input)prefix.push_back(prefix.back()+fixture_utf8_width(cp));Seq expected;for(auto position:icu_boundaries(input,static_cast<int>(mode)))expected.push_back(prefix.at(position));fixture_seq(row,expected);segment.add(row);
 }segment.finish();
 for(const auto& input:std::vector<Seq>{{0},{0xDF,0xFB03},{0x41,0x3A3},{0x49,0x307},{0x49,0x301},{0x69,0x307},{0x61,0,0x62}})fixture_cases(input,casing);casing.finish();
 std::ofstream bundle(dir/"current.bin",std::ios::binary);for(const auto& name:{norm.filename,segment.filename,casing.filename}){std::ifstream part(dir/name,std::ios::binary);if(!part)fail("smoke component");bundle<<part.rdbuf();}if(!bundle||!manifest)fail("smoke bundle");
 std::cout<<"UNICODE_SMOKE_FIXTURE_OK rows="<<norm.total+segment.total+casing.total<<"\n";
}

int main(int argc,char**argv){
 try{
  if(argc!=4||(std::string(argv[3])!="sample"&&std::string(argv[3])!="exhaustive")){std::cerr<<"usage: unicode_fixture_generate UCD_ROOT NEW_OUTPUT_DIRECTORY sample|exhaustive\n";return 64;}
  version_guard();std::filesystem::path root=argv[1],out=argv[2];std::ifstream readme(root/"ReadMe.txt");std::ostringstream readme_text;readme_text<<readme.rdbuf();if(readme_text.str().find("final data files for version 18.0.0")==std::string::npos)fail("expected final UCD18.0.0 ReadMe");if(!std::filesystem::create_directories(out))fail("output directory must be new");
  std::ofstream manifest(out/"MANIFEST.tsv");manifest<<"filename\tkind\trows\tpayload_bytes\n";
  FixtureWriter norm(out,"norm",1,manifest),segment(out,"segment",2,manifest),casing(out,"case",3,manifest);
  fixture_normalization(root,norm);norm.finish();fixture_segmentation(root,segment);segment.finish();fixture_casing(root,std::string(argv[3])=="exhaustive",casing);casing.finish();fixture_smoke(out);
  std::ofstream scope(out/"SCOPE.txt");scope<<"MODE "<<argv[3]<<"\nUnicode18.0.0 ICU79.1 CLDR49\nNormalization rows "<<norm.total<<"\nSegmentation rows "<<segment.total<<"\nCase/fold rows "<<casing.total<<"\n";if(!scope||!manifest)fail("fixture manifests");
  std::cout<<"UNICODE_FIXTURE_GENERATOR_OK mode="<<argv[3]<<" normalization="<<norm.total<<" segmentation="<<segment.total<<" case_fold="<<casing.total<<"\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"UNICODE_FIXTURE_GENERATOR_ERROR "<<e.what()<<"\n";return 1;}
}

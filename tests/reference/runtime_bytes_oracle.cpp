// C++23 independent byte oracle and control generator. No Sounio implementation is reused.
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace fs=std::filesystem;
using Bytes=std::vector<unsigned char>;
[[noreturn]] void fail(const std::string& s){throw std::runtime_error(s);}
Bytes read_bytes(const fs::path& path){
 std::ifstream in(path,std::ios::binary);
 if(!in)fail("unreadable source "+path.string());
 Bytes result;
 for(char ch;in.get(ch);)result.push_back(static_cast<unsigned char>(ch));
 if(!in.eof())fail("source read failure");
 return result;
}
template<std::size_t N> Bytes utf8_literal(const char8_t(&text)[N]){
 Bytes result;for(std::size_t i=0;i+1<N;++i)result.push_back(static_cast<unsigned char>(text[i]));return result;
}
void write_bytes(const fs::path& path,const Bytes& bytes){
 std::ofstream out(path,std::ios::binary);
 if(!bytes.empty())out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
 if(!out)fail("fixture write failure");
}
std::uint64_t number(std::string_view text){
 std::uint64_t value=0;const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
 if(error!=std::errc{}||end!=text.data()+text.size())fail("invalid nonnegative count");
 return value;
}
std::vector<std::string> lines(const fs::path& path){
 std::ifstream in(path);if(!in)fail("unreadable stdout");std::vector<std::string> result;
 for(std::string line;std::getline(in,line);)result.push_back(line);
 if(!in.eof())fail("stdout read failure");
 return result;
}
void write_lines(const fs::path& path,const std::vector<std::string>& data){
 std::ofstream out(path);for(const auto& line:data)out<<line<<'\n';if(!out)fail("control write failure");
}
void generate(const fs::path& root){
 if(fs::exists(root))fail("fresh fixture output required");fs::create_directories(root);
 Bytes all;for(unsigned int i=0;i<256;++i)all.push_back(static_cast<unsigned char>(i));
 write_bytes(root/"all-bytes.bin",all);
 write_bytes(root/"utf8-nul.bin",utf8_literal(u8"A\0\u00e9\U0001f4a1"));
 write_bytes(root/"split-scalar.bin",utf8_literal(u8"1234567\u00e9Z"));
 write_bytes(root/"invalid-utf8.bin",Bytes{0xc0,0xaf});
 write_bytes(root/"empty.bin",Bytes{});
 std::cout<<"RUNTIME_BYTES_FIXTURES_OK\n";
}
void mutate(std::string_view mode,const fs::path& input,const fs::path& output){
 auto data=lines(input);
 auto row=std::find_if(data.begin(),data.end(),[](const auto& s){return s.starts_with("RUNTIME_PROBE_BYTE ");});
 if(mode=="wrong-byte"||mode=="wrong-index"){
  if(row==data.end())fail("control needs byte row");
  std::istringstream in(row->substr(std::string_view("RUNTIME_PROBE_BYTE ").size()));std::uint64_t index=0,value=0;
  if(!(in>>index>>value)||value>255)fail("control row");
  if(mode=="wrong-byte")value=(value+1)%256;else ++index;
  *row="RUNTIME_PROBE_BYTE "+std::to_string(index)+" "+std::to_string(value);
 }else if(mode=="missing-row"){
  if(row==data.end())fail("control needs byte row");data.erase(row);
 }else if(mode=="duplicate-row"){
  if(row==data.end())fail("control needs byte row");const auto duplicate=*row;data.insert(row,duplicate);
 }else if(mode=="wrong-length"){
  auto length=std::find_if(data.begin(),data.end(),[](const auto& s){return s.find("_LEN ")!=std::string::npos;});
  if(length==data.end())fail("control needs length");const auto at=length->find("_LEN ")+5;
  *length=length->substr(0,at)+std::to_string(number(length->substr(at))+1);
 }else if(mode=="missing-after"){
  auto after=std::find_if(data.begin(),data.end(),[](const auto& s){return s.ends_with("_AFTER");});
  if(after==data.end())fail("control needs after marker");data.erase(after);
 }else fail("unknown mutation");
 if(fs::exists(output))fail("fresh control output required");write_lines(output,data);
}
void verify(std::string_view kind,const fs::path& source,const fs::path& stdout_path){
 Bytes expected;std::string label;
 bool owned=false;
 if(kind=="literal"){const std::string text="literal_probe";expected.assign(text.begin(),text.end());label="LITERAL";}
 else if(kind=="utf8-owned"){expected=utf8_literal(u8"A\0\u00e9\U0001f4a1");owned=true;}
 else{
  expected=read_bytes(source);
  if(kind=="binary")label="BINARY";
  else if(kind=="text")label="TEXT";
  else if(kind=="slice"){label="SLICE";expected.resize(std::min<std::size_t>(8,expected.size()));}
  else fail("unknown kind");
 }
 const auto data=lines(stdout_path);std::size_t row_count=0,start=0,length=0,after=0;
 const std::string row_prefix=owned?"BYTE ":"RUNTIME_PROBE_BYTE ";
 const std::string start_marker="RUNTIME_PROBE_"+label+"_START";
 const std::string length_prefix="RUNTIME_PROBE_"+label+"_LEN ";
 const std::string after_marker=owned?"UTF8_AS_BYTES_PROTO_OK":"RUNTIME_PROBE_"+label+"_AFTER";
 for(const auto& line:data){
  if(!owned&&line==start_marker){if(++start!=1||row_count!=0||length!=0||after!=0)fail("start order/count");}
  else if(!owned&&line.starts_with(length_prefix)){
   if(start!=1||++length!=1||row_count!=0||after!=0)fail("length order/count");
   if(number(std::string_view(line).substr(length_prefix.size()))!=expected.size())fail("length mismatch");
  }else if(line==after_marker){
   if(++after!=1||row_count!=expected.size()||(!owned&&(start!=1||length!=1)))fail("after order/count");
  }else if(line.starts_with(row_prefix)){
   if(after!=0||(!owned&&(start!=1||length!=1)))fail("byte order");
   std::istringstream in(line.substr(row_prefix.size()));std::uint64_t index=0,value=0;std::string extra;
   if(!(in>>index>>value)||(in>>extra)||value>255)fail("byte row shape");
   if(index!=row_count||row_count>=expected.size()||value!=expected[row_count])fail("byte mismatch at "+std::to_string(row_count));
   ++row_count;
  }else fail("unexpected stdout line: "+line);
 }
 if(after!=1||row_count!=expected.size()||(!owned&&(start!=1||length!=1)))fail("incomplete stdout");
 std::cout<<"RUNTIME_BYTES_ORACLE_OK kind="<<kind<<" bytes="<<expected.size()<<'\n';
}
int main(int argc,char** argv){
 try{
  if(argc==3&&std::string_view(argv[1])=="--generate"){generate(argv[2]);return 0;}
  if(argc==5&&std::string_view(argv[1])=="--mutate"){mutate(argv[2],argv[3],argv[4]);return 0;}
  if(argc!=7||std::string_view(argv[1])!="--kind"||std::string_view(argv[3])!="--source"||std::string_view(argv[5])!="--stdout")return 64;
  const std::string_view kind=argv[2];
  if(kind!="binary"&&kind!="literal"&&kind!="text"&&kind!="slice"&&kind!="utf8-owned")return 64;
  verify(kind,argv[4],argv[6]);return 0;
 }catch(const std::exception& e){std::cerr<<"RUNTIME_BYTES_ORACLE_ERROR "<<e.what()<<'\n';return 1;}
}

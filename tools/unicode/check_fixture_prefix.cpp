// C++23 preservation gate: every original serialized row must remain unchanged.
#define SOUNIO_CHUNK_RUNNER_NO_MAIN
#include "run_chunks.cpp"
#include <iterator>
struct RowStream {
 fs::path root;std::vector<Fixture> files;std::size_t file=0;std::uint32_t row=0;std::vector<unsigned char> bytes;std::size_t at=24;
 explicit RowStream(fs::path p):root(std::move(p)),files(manifest(root)){}
 std::uint32_t u32(std::size_t p){if(p>bytes.size()||bytes.size()-p<4)fail("row u32 bounds");return std::uint32_t(bytes[p])|(std::uint32_t(bytes[p+1])<<8)|(std::uint32_t(bytes[p+2])<<16)|(std::uint32_t(bytes[p+3])<<24);}
 bool next(std::vector<unsigned char>&out){if(bytes.empty()||row==files[file-1].rows){if(file==files.size())return false;if(!bytes.empty()&&at!=bytes.size())fail("trailing row bytes");std::ifstream f(root/files[file].name,std::ios::binary);bytes.assign(std::istreambuf_iterator<char>(f),{});if(files[file].kind!=4)fail("prefix only supports collation rows");++file;row=0;at=24;}auto begin=at;if(at>bytes.size()||bytes.size()-at<20)fail("row metadata bounds");at+=20;for(int s=0;s<2;++s){auto n=u32(at);at+=4;if(n>(bytes.size()-at)/4)fail("row sequence bounds");at+=std::size_t(n)*4;}out.assign(bytes.begin()+begin,bytes.begin()+at);++row;return true;}
};
int main(int argc,char**argv){try{if(argc!=3)return 64;RowStream before(argv[1]),after(argv[2]);std::vector<unsigned char>a,b;N rows=0;while(before.next(a)){if(!after.next(b)||a!=b)fail("original row changed at "+std::to_string(rows));++rows;}if(!rows)fail("empty original prefix");std::cout<<"FIXTURE_PREFIX_PRESERVED rows="<<rows<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"FIXTURE_PREFIX_ERROR "<<e.what()<<"\n";return 1;}}

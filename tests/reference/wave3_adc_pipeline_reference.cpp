#include <cstring>
#include <arrow/util/config.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <thread>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
// Independent C++23 exact ADC oracle and Arrow C++26 IPC consumer.
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static_assert(ARROW_VERSION_MAJOR==26&&ARROW_VERSION_MINOR==0&&ARROW_VERSION_PATCH==0);
using I=std::int64_t;
static_assert(static_cast<__int128>(1000000)*1000000*1000000<=INT64_MAX);
namespace fs=std::filesystem;
static void need(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
template<class T>static T take(arrow::Result<T> r){need(r.ok(),r.status().ToString());return std::move(r).ValueOrDie();}
static I integer(std::string text){
 if(!text.empty()&&text.front()=='"'){need(text.size()>=2&&text.back()=='"',"bad quoted integer");text=text.substr(1,text.size()-2);}
 if(!text.empty()&&text.front()=='+')text.erase(0,1);
 I x=0;const auto [p,e]=std::from_chars(text.data(),text.data()+text.size(),x);
 need(e==std::errc{}&&p==text.data()+text.size(),"invalid integer");return x;
}
static std::vector<I> samples(const fs::path& input){
 std::ifstream in(input,std::ios::binary);need(bool(in),"open input");
 std::string line;need(bool(std::getline(in,line)),"header");
 if(line.starts_with("\xef\xbb\xbf"))line.erase(0,3);
 if(line.ends_with('\r'))line.pop_back();
 need(line=="sample_id,amplitude_adc","CSV header");
 std::vector<I> values;
 while(std::getline(in,line)){
  if(line.ends_with('\r'))line.pop_back();
  const auto comma=line.find(',');need(comma!=std::string::npos&&line.find(',',comma+1)==std::string::npos,"CSV columns");
  need(integer(line.substr(0,comma))==static_cast<I>(values.size()),"CSV sample order");
  const I x=integer(line.substr(comma+1));need(x>=-1000000&&x<=1000000,"ADC domain");values.push_back(x);
 }
 need(!values.empty()&&values.size()<=1000000,"sample cardinality");return values;
}
static void generate(const fs::path& dir,I count){
 need(count>0&&count<=1000000,"generate count");fs::create_directories(dir);
 std::ofstream out(dir/"input.csv",std::ios::binary);out<<"\xef\xbb\xbfsample_id,amplitude_adc\r\n";
 for(I i=0;i<count;++i){
  I x=(i*7919)%2000001-1000000;
  if(i%17==0)x=0;
  if(i%19==0)x=1000000;
  if(i%23==0)x=-1000000;
  out<<i<<',';
  if(i%3==0)out<<'"'<<(x>=0?"+":"")<<x<<'"';else out<<x;
  if(i+1<count)out<<(i%2==0?"\r\n":"\n");
 }
 need(bool(out),"fixture write");
}
static void verify(const fs::path& input,const fs::path& output,const std::string& mutation){
 const auto expected=samples(input);

 std::ifstream wire(output,std::ios::binary);
 need(bool(wire),"open Arrow bytes");
 wire.seekg(-8,std::ios::end);
 char tail[8]{};
 wire.read(tail,8);
 const unsigned char eos[8]={255,255,255,255,0,0,0,0};
 need(wire.gcount()==8&&std::memcmp(tail,eos,8)==0,"explicit Arrow EOS missing");

 auto file=take(arrow::io::ReadableFile::Open(output.string()));
 auto reader=take(arrow::ipc::RecordBatchStreamReader::Open(file));
 const auto schema=reader->schema();
 need(schema->num_fields()==3,"Arrow columns");
 const char* names[]={"sample_id","amplitude_adc","squared_amplitude"};
 for(int c=0;c<3;++c)need(schema->field(c)->name()==names[c]&&schema->field(c)->type()->Equals(arrow::int64())&&!schema->field(c)->nullable(),"Arrow schema");
 I row=0,batches=0;__int128 sum=0;
 for(;;){
  std::shared_ptr<arrow::RecordBatch> batch;const auto status=reader->ReadNext(&batch);
  need(status.ok(),status.ToString());if(!batch)break;
  need(batch->num_rows()>0&&batch->num_rows()<=16&&batch->num_columns()==3,"batch bounds");
  for(int c=0;c<3;++c)need(batch->column(c)->null_count()==0,"unexpected null");
  auto ids=std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
  auto amps=std::static_pointer_cast<arrow::Int64Array>(batch->column(1));
  auto squares=std::static_pointer_cast<arrow::Int64Array>(batch->column(2));
  for(I r=0;r<batch->num_rows();++r,++row){
   need(row<static_cast<I>(expected.size()),"extra row");
   I id=ids->Value(r),value=squares->Value(r);
   if(row==0&&mutation=="wrong-value")++value;
   if(row<2&&mutation=="wrong-order")id=1-row;
   const __int128 square=static_cast<__int128>(expected[static_cast<std::size_t>(row)])*expected[static_cast<std::size_t>(row)];
   need(id==row&&amps->Value(r)==expected[static_cast<std::size_t>(row)]&&value==square,"Arrow row "+std::to_string(row));
   sum+=square;
  }
  ++batches;
 }
 need(row==static_cast<I>(expected.size()),"missing rows");
 need(take(file->Tell())==take(file->GetSize()),"trailing Arrow bytes");
 need(sum<=INT64_MAX,"sum bound");
 std::cout<<"ADC_ARROW_CPP_OK rows="<<row<<" batches="<<batches<<" sum_squares="<<static_cast<I>(sum)<<'\n';
}

static void mutate(const fs::path& input,const fs::path& output,const std::string& kind){
 auto file=take(arrow::io::ReadableFile::Open(input.string()));
 auto reader=take(arrow::ipc::RecordBatchStreamReader::Open(file));
 auto sink=take(arrow::io::FileOutputStream::Open(output.string()));
 auto writer=take(arrow::ipc::MakeStreamWriter(sink,reader->schema()));
 bool first=true;
 for(;;){
  std::shared_ptr<arrow::RecordBatch> batch;need(reader->ReadNext(&batch).ok(),"mutation read");if(!batch)break;
  if(first){
   const int col=kind=="wrong-value"?2:0;
   auto old=std::static_pointer_cast<arrow::Int64Array>(batch->column(col));
   arrow::Int64Builder builder;
   for(I i=0;i<old->length();++i){
    I value=old->Value(i);
    if(kind=="wrong-value"&&i==0)++value;
    if(kind=="wrong-order"&&i<2)value=1-i;
    need(builder.Append(value).ok(),"mutation append");
   }
   auto columns=batch->columns();columns[static_cast<std::size_t>(col)]=take(builder.Finish());
   batch=arrow::RecordBatch::Make(reader->schema(),batch->num_rows(),columns);first=false;
  }
  need(writer->WriteRecordBatch(*batch).ok(),"mutation write");
 }
 need(writer->Close().ok()&&sink->Close().ok(),"mutation close");
}


static void peer(const std::string& port_text,const fs::path& input,I chunk,I delay_ms,const std::string& mode){
 const I port=integer(port_text);
 need(port>0&&port<=65535&&chunk>0&&chunk<=65536&&delay_ms>=0&&delay_ms<=1000,"peer parameters");
 std::ifstream file(input,std::ios::binary);need(bool(file),"peer input");
 const std::string data((std::istreambuf_iterator<char>(file)),{});
 const int fd=::socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);need(fd>=0,"peer socket");
 struct Close{int fd;~Close(){if(fd>=0)::close(fd);}} close{fd};
 int one=1,send_bytes=4096;timeval timeout{5,0};
 need(setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one))==0,"TCP_NODELAY");
 need(setsockopt(fd,SOL_SOCKET,SO_SNDBUF,&send_bytes,sizeof(send_bytes))==0,"SO_SNDBUF");
 need(setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout))==0,"SO_SNDTIMEO");
 sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(static_cast<std::uint16_t>(port));address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
 need(::connect(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0,"peer connect");
 std::size_t sent=0,calls=0;
 while(sent<data.size()){
  const auto amount=std::min(static_cast<std::size_t>(chunk),data.size()-sent);
  const auto n=::send(fd,data.data()+sent,amount,MSG_NOSIGNAL);
  if(n<0&&errno==EINTR)continue;
  need(n>0,"TCP_SEND_ERROR errno="+std::to_string(errno));
  sent+=static_cast<std::size_t>(n);++calls;
  if(mode=="slow")std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
 }
 if(mode=="hold")std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
 // The receiver may intentionally close on a tested error before shutdown.
 ::shutdown(fd,SHUT_WR);
 std::cout<<"TCP_PEER_SENT bytes="<<sent<<" calls="<<calls<<'\n';
}


static I rss_kib(const fs::path& path){
 std::ifstream file(path);need(bool(file),"resource receipt");
 std::string line;
 const std::string key="Maximum resident set size (kbytes): ";
 while(std::getline(file,line)){
  const auto at=line.find(key);
  if(at!=std::string::npos)return integer(line.substr(at+key.size()));
 }
 throw std::runtime_error("missing max RSS");
}
static void resources(const fs::path& small,const fs::path& scale,const fs::path& tcp){
 const I a=rss_kib(small),b=rss_kib(scale),c=rss_kib(tcp);
 need(a>0&&b>0&&c>0,"nonpositive RSS");
 need(b<=131072&&c<=131072,"128MiB resident-memory limit");
 need(b-a<=16384&&c-a<=16384,"RSS growth exceeds 16MiB over small fixture");
 std::cout<<"ADC_RSS_CPP_OK small_kib="<<a<<" scale_kib="<<b<<" tcp_kib="<<c<<'\n';
}

int main(int argc,char** argv){
 try{
  need(argc>=2,"mode");
  if(std::string(argv[1])=="generate"){need(argc==4,"generate DIR COUNT");generate(argv[2],integer(argv[3]));}
  else if(std::string(argv[1])=="resources"){need(argc==5,"resources SMALL SCALE TCP");resources(argv[2],argv[3],argv[4]);}
  else if(std::string(argv[1])=="peer"){need(argc==7,"peer PORT CSV CHUNK DELAY_MS MODE");peer(argv[2],argv[3],integer(argv[4]),integer(argv[5]),argv[6]);}
  else if(std::string(argv[1])=="mutate"){need(argc==5,"mutate INPUT OUTPUT KIND");mutate(argv[2],argv[3],argv[4]);}
  else{need((argc==4||argc==5)&&std::string(argv[1])=="verify","verify CSV ARROW [mutation]");verify(argv[2],argv[3],argc==5?argv[4]:"");}
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

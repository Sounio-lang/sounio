// Independent C++23 long-double normal oracle and Arrow26 IPC consumer.
#include <arrow/util/config.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <cerrno>
#include <chrono>
#include <thread>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
static_assert(ARROW_VERSION_MAJOR==26&&ARROW_VERSION_MINOR==0&&ARROW_VERSION_PATCH==0);
static_assert(std::numeric_limits<long double>::digits>=64);
using I=std::int64_t;
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
 need(line=="sample_id,z_milli","CSV header");
 std::vector<I> values;
 while(std::getline(in,line)){
  if(line.ends_with('\r'))line.pop_back();
  const auto comma=line.find(',');need(comma!=std::string::npos&&line.find(',',comma+1)==std::string::npos,"CSV columns");
  need(integer(line.substr(0,comma))==static_cast<I>(values.size()),"CSV sample order");
  const I x=integer(line.substr(comma+1));need(x>=-40000&&x<=40000,"normal domain");values.push_back(x);
 }
 need(!values.empty()&&values.size()<=1000000,"sample cardinality");return values;
}
static I fixture(I i){
 static constexpr I edge[]={0,1,-1,500,-500,1000,-1000,2000,-2000,5000,-5000,8000,-8000,10000,-10000,20000,-20000,38000,-38000,40000,-40000};
 i%=257;
 if(i<static_cast<I>(std::size(edge)))return edge[i];
 return (i*311)%80001-40000;
}
static void generate(const fs::path& dir,I count){
 need(count>0&&count<=1000000,"generate count");fs::create_directories(dir);
 std::ofstream out(dir/"input.csv",std::ios::binary);out<<"\xef\xbb\xbfsample_id,z_milli\r\n";
 for(I i=0;i<count;++i){
  const I x=fixture(i);out<<i<<',';
  if(i%3==0)out<<'"'<<(x>=0?"+":"")<<x<<'"';else out<<x;
  if(i+1<count)out<<(i%2==0?"\r\n":"\n");
 }
 need(bool(out),"fixture write");
}
static std::array<long double,4> reference(I milli){
 // The application first rounds milli/1000 to binary64. Match that input
 // exactly; all subsequent independent erfc/log1p calculations are extended.
 const double x=static_cast<double>(milli)/1000.0;
 const long double z=static_cast<long double>(x);
 const long double tail=std::erfc(std::abs(z)/std::sqrt(2.0L))/2.0L;
 need(tail>0&&tail<=0.5L,"long-double erfc range");
 const long double c=z<=0?tail:1.0L-tail,s=z>=0?tail:1.0L-tail;
 const long double lc=z<=0?std::log(tail):std::log1p(-tail);
 const long double ls=z>=0?std::log(tail):std::log1p(-tail);
 return {c,s,lc,ls};
}
static void concordance(double actual,long double exact,I row,int col){
 const double expected=static_cast<double>(exact);
 need(std::isfinite(actual),"nonfinite normal column row="+std::to_string(row));
 if(col<4)need(actual>=0&&actual<=1&&!std::signbit(actual),"probability range");
 else need(actual<=0,"log-probability range");
 const long double floor=8.0L*static_cast<long double>(std::numeric_limits<double>::denorm_min());
 const long double error=std::abs(static_cast<long double>(actual)-static_cast<long double>(expected));
 const long double limit=std::max(floor,2.0e-12L*std::abs(static_cast<long double>(expected)));
 need(error<=limit,"normal oracle row="+std::to_string(row)+" col="+std::to_string(col)+" actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
static void verify(const fs::path& input,const fs::path& output){
 const auto expected=samples(input);
 std::ifstream wire(output,std::ios::binary);need(bool(wire),"open Arrow bytes");
 wire.seekg(-8,std::ios::end);char tail[8]{};wire.read(tail,8);
 const unsigned char eos[8]={255,255,255,255,0,0,0,0};
 need(wire.gcount()==8&&std::memcmp(tail,eos,8)==0,"explicit Arrow EOS missing");
 auto file=take(arrow::io::ReadableFile::Open(output.string()));
 auto reader=take(arrow::ipc::RecordBatchStreamReader::Open(file));
 const auto schema=reader->schema();
 need(schema->num_fields()==6,"Arrow columns");
 const char* names[]={"sample_id","z_milli","cdf","sf","logcdf","logsf"};
 for(int c=0;c<6;++c){
  const auto type=c<2?arrow::int64():arrow::float64();
  need(schema->field(c)->name()==names[c]&&schema->field(c)->type()->Equals(type)&&!schema->field(c)->nullable(),"Arrow schema");
 }
 std::map<I,std::array<long double,4>> cache;
 I row=0,batches=0;
 for(;;){
  std::shared_ptr<arrow::RecordBatch> batch;const auto status=reader->ReadNext(&batch);
  need(status.ok(),status.ToString());if(!batch)break;
  need(batch->num_rows()>0&&batch->num_rows()<=16&&batch->num_columns()==6,"batch bounds");
  for(int c=0;c<6;++c)need(batch->column(c)->null_count()==0,"unexpected null");
  auto ids=std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
  auto milli=std::static_pointer_cast<arrow::Int64Array>(batch->column(1));
  for(I r=0;r<batch->num_rows();++r,++row){
   need(row<static_cast<I>(expected.size()),"extra row");
   const I z=expected[static_cast<std::size_t>(row)];
   need(ids->Value(r)==row&&milli->Value(r)==z,"Arrow row order");
   if(!cache.contains(z))cache.emplace(z,reference(z));
   for(int c=2;c<6;++c){
    auto values=std::static_pointer_cast<arrow::DoubleArray>(batch->column(c));
    concordance(values->Value(r),cache.at(z)[static_cast<std::size_t>(c-2)],row,c);
   }
  }
  ++batches;
 }
 need(row==static_cast<I>(expected.size()),"missing rows");
 need(take(file->Tell())==take(file->GetSize()),"trailing Arrow bytes");
 std::cout<<"NORMAL_ARROW_CPP_OK rows="<<row<<" batches="<<batches<<" distinct_z="<<cache.size()<<'\n';
}
static void mutate(const fs::path& input,const fs::path& output,const std::string& kind){
 need(kind=="wrong-value"||kind=="wrong-order","mutation kind");
 auto file=take(arrow::io::ReadableFile::Open(input.string()));
 auto reader=take(arrow::ipc::RecordBatchStreamReader::Open(file));
 auto sink=take(arrow::io::FileOutputStream::Open(output.string()));
 auto writer=take(arrow::ipc::MakeStreamWriter(sink,reader->schema()));
 bool first=true;
 for(;;){
  std::shared_ptr<arrow::RecordBatch> batch;need(reader->ReadNext(&batch).ok(),"mutation read");if(!batch)break;
  if(first){
   auto columns=batch->columns();
   if(kind=="wrong-value"){
    auto old=std::static_pointer_cast<arrow::DoubleArray>(batch->column(2));arrow::DoubleBuilder builder;
    for(I i=0;i<old->length();++i)need(builder.Append(i==0?0.123:old->Value(i)).ok(),"mutation append");
    columns[2]=take(builder.Finish());
   }else{
    auto old=std::static_pointer_cast<arrow::Int64Array>(batch->column(0));arrow::Int64Builder builder;
    for(I i=0;i<old->length();++i)need(builder.Append(i<2?1-i:old->Value(i)).ok(),"mutation append");
    columns[0]=take(builder.Finish());
   }
   batch=arrow::RecordBatch::Make(reader->schema(),batch->num_rows(),columns);first=false;
  }
  need(writer->WriteRecordBatch(*batch).ok(),"mutation write");
 }
 need(writer->Close().ok()&&sink->Close().ok(),"mutation close");
}

static void oracle_fixture(const fs::path& dir,I count){
 generate(dir,count);const auto data=samples(dir/"input.csv");
 auto schema=arrow::schema({arrow::field("sample_id",arrow::int64(),false),arrow::field("z_milli",arrow::int64(),false),
  arrow::field("cdf",arrow::float64(),false),arrow::field("sf",arrow::float64(),false),
  arrow::field("logcdf",arrow::float64(),false),arrow::field("logsf",arrow::float64(),false)});
 auto sink=take(arrow::io::FileOutputStream::Open((dir/"oracle.arrow").string()));
 auto writer=take(arrow::ipc::MakeStreamWriter(sink,schema));
 std::map<I,std::array<long double,4>> cache;
 for(I base=0;base<count;base+=16){
  arrow::Int64Builder ids,zs;std::array<arrow::DoubleBuilder,4> results;
  const I n=std::min<I>(16,count-base);
  for(I i=0;i<n;++i){
   const I z=data[static_cast<std::size_t>(base+i)];
   need(ids.Append(base+i).ok()&&zs.Append(z).ok(),"oracle fixture append");
   if(!cache.contains(z))cache.emplace(z,reference(z));
   for(int c=0;c<4;++c)need(results[static_cast<std::size_t>(c)].Append(static_cast<double>(cache.at(z)[static_cast<std::size_t>(c)])).ok(),"oracle fixture probability");
  }
  std::vector<std::shared_ptr<arrow::Array>> cols{take(ids.Finish()),take(zs.Finish())};
  for(auto& builder:results)cols.push_back(take(builder.Finish()));
  auto batch=arrow::RecordBatch::Make(schema,n,cols);
  need(writer->WriteRecordBatch(*batch).ok(),"oracle fixture batch");
 }
 need(writer->Close().ok()&&sink->Close().ok(),"oracle fixture close");
 verify(dir/"input.csv",dir/"oracle.arrow");
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
 std::cout<<"NORMAL_RSS_CPP_OK small_kib="<<a<<" scale_kib="<<b<<" tcp_kib="<<c<<'\n';
}


int main(int argc,char** argv){
 try{
  need(argc>=2,"mode");
  const std::string mode=argv[1];
  if(mode=="oracle-fixture"){need(argc==4,"oracle-fixture DIR COUNT");oracle_fixture(argv[2],integer(argv[3]));}
  else if(mode=="generate"){need(argc==4,"generate DIR COUNT");generate(argv[2],integer(argv[3]));}
  else if(mode=="resources"){need(argc==5,"resources SMALL SCALE TCP");resources(argv[2],argv[3],argv[4]);}
  else if(mode=="peer"){need(argc==7,"peer PORT CSV CHUNK DELAY_MS MODE");peer(argv[2],argv[3],integer(argv[4]),integer(argv[5]),argv[6]);}
  else if(mode=="mutate"){need(argc==5,"mutate INPUT OUTPUT KIND");mutate(argv[2],argv[3],argv[4]);}
  else {need(argc==4&&mode=="verify","verify CSV ARROW");verify(argv[2],argv[3]);}
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

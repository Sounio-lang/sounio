// Independent Arrow C++26 interoperability judge. No Python/NumPy/PyArrow.
// Preserves scripts/arrow_{ipc,spec,ext,flight}_check.py cases; see inventory.
#include <arrow/api.h>
#include <arrow/config.h>
#include <arrow/extension_type.h>
#include <arrow/flight/types.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/util/config.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
static_assert(ARROW_VERSION_MAJOR == 26 && ARROW_VERSION_MINOR == 0 && ARROW_VERSION_PATCH == 0);
namespace fs=std::filesystem;
using Bytes=std::vector<std::uint8_t>;
using A=std::shared_ptr<arrow::Array>;
struct Bad:std::runtime_error {using std::runtime_error::runtime_error;};
void need(bool ok,const std::string& why){if(!ok)throw Bad(why);}
template<class T>T take(arrow::Result<T> r){if(!r.ok())throw Bad(r.status().ToString());return std::move(r).ValueOrDie();}
void okay(const arrow::Status& s){if(!s.ok())throw Bad(s.ToString());}
void passed(std::string_view id){std::cout<<"CASE_OK "<<id<<"\n";}
Bytes bytes(const fs::path& p){
 std::ifstream f(p,std::ios::binary);need(bool(f),"open "+p.string());
 return Bytes(std::istreambuf_iterator<char>(f),{});
}
Bytes hex(std::string_view s){need(s.size()%2==0,"hex width");Bytes b;for(std::size_t i=0;i<s.size();i+=2)b.push_back(static_cast<std::uint8_t>(std::stoul(std::string(s.substr(i,2)),nullptr,16)));return b;}
std::uint32_t u32(const Bytes& b,std::size_t at,bool big=false){
 need(at<=b.size()&&b.size()-at>=4,"short u32");std::uint32_t n=0;
 for(unsigned i=0;i<4;++i){n|=std::uint32_t(b[at+i])<<(8*(big?3-i:i));}return n;
}
void le32(Bytes& b,std::uint32_t v){for(unsigned i=0;i<4;++i)b.push_back(static_cast<std::uint8_t>(v>>(8*i)));}
bool has(const Bytes& b,const Bytes& p){return std::search(b.begin(),b.end(),p.begin(),p.end())!=b.end();}
void framing(const fs::path& p,bool file,bool needles){
 auto b=bytes(p);
 if(file){
  const Bytes magic{'A','R','R','O','W','1'};
  need(b.size()>=18&&std::equal(magic.begin(),magic.end(),b.begin())&&std::equal(magic.begin(),magic.end(),b.end()-6),"file magic");
  auto n=u32(b,b.size()-10);need(n>0&&n<=b.size()-18,"footer size");
  const auto at=b.size()-10-n-8;need(at>=8&&u32(b,at)==0xffffffffU&&u32(b,at+4)==0,"eos before footer");
 }else{need(b.size()>=8&&u32(b,0)==0xffffffffU&&u32(b,b.size()-8)==0xffffffffU&&u32(b,b.size()-4)==0,"stream continuation/EOS");}
 if(needles)for(auto s:{"fdffffffffffffff","000000000000f83f","00000000000002c0","616263"})need(has(b,hex(s)),"structural value needle");
}
struct Stream {std::shared_ptr<arrow::Schema> schema;std::vector<std::shared_ptr<arrow::RecordBatch>> batches;};
Stream stream_blob(const Bytes& b){
 auto in=std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(std::string(b.begin(),b.end())));
 auto reader=take(arrow::ipc::RecordBatchStreamReader::Open(in));Stream s{reader->schema(),{}};
 for(;;){std::shared_ptr<arrow::RecordBatch> batch;okay(reader->ReadNext(&batch));if(!batch)break;okay(batch->ValidateFull());s.batches.push_back(std::move(batch));}return s;
}
Stream stream(const fs::path& p){return stream_blob(bytes(p));}
Stream file(const fs::path& p){
 auto in=take(arrow::io::ReadableFile::Open(p.string()));auto reader=take(arrow::ipc::RecordBatchFileReader::Open(in));Stream s{reader->schema(),{}};
 for(int i=0;i<reader->num_record_batches();++i){auto b=take(reader->ReadRecordBatch(i));okay(b->ValidateFull());s.batches.push_back(std::move(b));}return s;
}
A storage(A a){while(a->type_id()==arrow::Type::EXTENSION)a=std::static_pointer_cast<arrow::ExtensionArray>(a)->storage();return a;}
std::int64_t integer_scalar(const std::shared_ptr<arrow::Scalar>& s){
 need(s->is_valid,"null integer");
 switch(s->type->id()){
#define INT_CASE(ID,TYPE) case arrow::Type::ID:return static_cast<const arrow::TYPE&>(*s).value;
 INT_CASE(INT8,Int8Scalar) INT_CASE(INT16,Int16Scalar) INT_CASE(INT32,Int32Scalar) INT_CASE(INT64,Int64Scalar)
 INT_CASE(UINT8,UInt8Scalar) INT_CASE(UINT16,UInt16Scalar) INT_CASE(UINT32,UInt32Scalar)
 INT_CASE(DATE32,Date32Scalar) INT_CASE(DATE64,Date64Scalar) INT_CASE(TIME32,Time32Scalar) INT_CASE(TIME64,Time64Scalar)
 INT_CASE(TIMESTAMP,TimestampScalar) INT_CASE(DURATION,DurationScalar) INT_CASE(INTERVAL_MONTHS,MonthIntervalScalar)
#undef INT_CASE
 case arrow::Type::UINT64:{const auto v=static_cast<const arrow::UInt64Scalar&>(*s).value;need(v<=INT64_MAX,"uint64 range");return static_cast<std::int64_t>(v);}
 case arrow::Type::SPARSE_UNION:case arrow::Type::DENSE_UNION:return integer_scalar(static_cast<const arrow::UnionScalar&>(*s).child_value());
 default:throw Bad("not integer scalar: "+s->type->ToString());
 }
}
std::int64_t integer(A a,std::int64_t i){
 a=storage(a);need(i>=0&&i<a->length(),"integer index");
 if(a->type_id()==arrow::Type::DICTIONARY){auto d=std::static_pointer_cast<arrow::DictionaryArray>(a);return integer(d->dictionary(),integer(d->indices(),i));}
 if(a->type_id()==arrow::Type::RUN_END_ENCODED){auto r=std::static_pointer_cast<arrow::RunEndEncodedArray>(a);for(std::int64_t j=0;j<r->run_ends()->length();++j)if(integer(r->run_ends(),j)>i+a->offset())return integer(r->values(),j);throw Bad("run ends");}
 return integer_scalar(take(a->GetScalar(i)));
}
double real(A a,std::int64_t i){
 a=storage(a);auto s=take(a->GetScalar(i));need(s->is_valid,"null real");
 if(s->type->id()==arrow::Type::DOUBLE)return static_cast<const arrow::DoubleScalar&>(*s).value;
 if(s->type->id()==arrow::Type::FLOAT)return static_cast<const arrow::FloatScalar&>(*s).value;
 throw Bad("not real");
}
std::string text(A a,std::int64_t i){
 a=storage(a);need(i>=0&&i<a->length()&&!a->IsNull(i),"null/text index");
 if(a->type_id()==arrow::Type::DICTIONARY){auto d=std::static_pointer_cast<arrow::DictionaryArray>(a);return text(d->dictionary(),integer(d->indices(),i));}
 switch(a->type_id()){
#define TEXT_CASE(ID,TYPE) case arrow::Type::ID:return std::string(std::static_pointer_cast<arrow::TYPE>(a)->GetView(i));
 TEXT_CASE(STRING,StringArray) TEXT_CASE(BINARY,BinaryArray) TEXT_CASE(LARGE_STRING,LargeStringArray)
 TEXT_CASE(LARGE_BINARY,LargeBinaryArray) TEXT_CASE(FIXED_SIZE_BINARY,FixedSizeBinaryArray)
 TEXT_CASE(STRING_VIEW,StringViewArray) TEXT_CASE(BINARY_VIEW,BinaryViewArray)
#undef TEXT_CASE
 default:throw Bad("not bytes: "+a->type()->ToString());
 }
}
A list(A a,std::int64_t i){
 auto scalar=take(storage(a)->GetScalar(i));need(scalar->is_valid,"null list");
 auto l=std::dynamic_pointer_cast<arrow::BaseListScalar>(scalar);need(bool(l),"not list");return l->value;
}
A col(const std::shared_ptr<arrow::RecordBatch>& b,std::string_view n){auto a=b->GetColumnByName(std::string(n));need(bool(a),"missing column "+std::string(n));return a;}
std::shared_ptr<arrow::Field> field(const Stream& s,std::string_view n){auto f=s.schema->GetFieldByName(std::string(n));need(bool(f),"missing field");return f;}
void ints(A a,std::initializer_list<std::int64_t> want,std::string_view name){
 need(a->length()==static_cast<std::int64_t>(want.size()),std::string(name)+" length");std::int64_t i=0;for(auto x:want){need(integer(a,i)==x,std::string(name)+"["+std::to_string(i)+"]");++i;}
}
void texts(A a,std::initializer_list<std::string_view> want,std::string_view name){
 need(a->length()==static_cast<std::int64_t>(want.size()),std::string(name)+" length");std::int64_t i=0;for(auto x:want){need(text(a,i)==x,std::string(name)+"["+std::to_string(i)+"]");++i;}
}
void basic(const Stream& s){
 need(s.batches.size()==1,"basic batch count");auto b=s.batches[0];need(b->num_columns()==7&&b->num_rows()==3,"basic shape");
 ints(col(b,"id"),{1,2,-3},"id");auto x=col(b,"x");need(real(x,0)==1.5&&x->IsNull(1)&&real(x,2)==-2.25,"x values");
 auto ok=std::static_pointer_cast<arrow::BooleanArray>(col(b,"ok"));need(ok->Value(0)&&!ok->Value(1)&&ok->IsNull(2),"ok values");
 auto n=col(b,"name");need(text(n,0)=="ab"&&n->IsNull(1)&&text(n,2)=="c","name values");
 ints(col(b,"tiny"),{-1,0,127},"tiny");ints(col(b,"code"),{65535,1,256},"code");need(col(b,"z")->null_count()==3,"null values");
}
std::string meta(const std::shared_ptr<arrow::Field>& f,std::string_view key){need(bool(f->metadata()),"missing field metadata");return take(f->metadata()->Get(std::string(key)));}
std::shared_ptr<arrow::ExtensionType> ext(const Stream& s,std::string_view n,std::string_view want){
 auto t=field(s,n)->type();need(t->id()==arrow::Type::EXTENSION,"extension not recognized "+std::string(n));
 auto e=std::static_pointer_cast<arrow::ExtensionType>(t);need(e->extension_name()==want,"extension name");return e;
}
// Metadata JSON is checked independently; only integer metadata numbers occur here.
struct Json {
 using Array=std::vector<Json>;using Object=std::map<std::string,Json>;
 std::variant<std::nullptr_t,bool,std::int64_t,std::string,Array,Object> value;
};
class JsonReader {
 std::string_view s;std::size_t p=0;
 void space(){while(p<s.size()&&(s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t'))++p;}
 char next(){need(p<s.size(),"JSON truncated");return s[p++];}
 static void utf8(std::string& out,unsigned cp){
  if(cp<128)out+=static_cast<char>(cp);else if(cp<2048){out+=static_cast<char>(192+(cp>>6));out+=static_cast<char>(128+(cp&63));}
  else if(cp<65536){out+=static_cast<char>(224+(cp>>12));out+=static_cast<char>(128+((cp>>6)&63));out+=static_cast<char>(128+(cp&63));}
  else{out+=static_cast<char>(240+(cp>>18));out+=static_cast<char>(128+((cp>>12)&63));out+=static_cast<char>(128+((cp>>6)&63));out+=static_cast<char>(128+(cp&63));}
 }
 unsigned hex4(){unsigned x=0;for(int i=0;i<4;++i){char c=next();need((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F'),"JSON hex");x=x*16+(c<='9'?c-'0':c<='F'?c-'A'+10:c-'a'+10);}return x;}
 std::string string(){
  need(next()=='"',"JSON string");std::string out;
  for(;;){char c=next();if(c=='"')return out;need(static_cast<unsigned char>(c)>=32,"JSON control");if(c!='\\'){out+=c;continue;}
   c=next();switch(c){case '"':case '\\':case '/':out+=c;break;case 'b':out+='\b';break;case 'f':out+='\f';break;case 'n':out+='\n';break;case 'r':out+='\r';break;case 't':out+='\t';break;
   case 'u':{unsigned x=hex4();if(x>=0xd800&&x<=0xdbff){need(next()=='\\'&&next()=='u',"JSON surrogate");auto lo=hex4();need(lo>=0xdc00&&lo<=0xdfff,"JSON surrogate");x=0x10000+(x-0xd800)*1024+lo-0xdc00;}else need(!(x>=0xdc00&&x<=0xdfff),"JSON low surrogate");utf8(out,x);break;}default:throw Bad("JSON escape");}
  }
 }
 Json read(){
  space();need(p<s.size(),"JSON empty");char c=s[p];
  if(c=='"')return Json{string()};
  if(c=='['){++p;Json::Array a;space();if(p<s.size()&&s[p]==']'){++p;return Json{a};}for(;;){a.push_back(read());space();c=next();if(c==']')return Json{a};need(c==',',"JSON array");}}
  if(c=='{'){++p;Json::Object o;space();if(p<s.size()&&s[p]=='}'){++p;return Json{o};}for(;;){space();auto k=string();space();need(next()==':',"JSON colon");need(o.emplace(k,read()).second,"JSON duplicate");space();c=next();if(c=='}')return Json{o};need(c==',',"JSON object");}}
  for(auto [word,value]:{std::pair{"true",Json{true}},std::pair{"false",Json{false}},std::pair{"null",Json{nullptr}}})if(s.substr(p,std::string_view(word).size())==word){p+=std::string_view(word).size();return value;}
  auto begin=p;if(s[p]=='-')++p;while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;need(p>begin,"JSON number");return Json{std::stoll(std::string(s.substr(begin,p-begin)))};
 }
 public:explicit JsonReader(std::string_view input):s(input){}Json parse(){auto v=read();space();need(p==s.size(),"JSON trailing");return v;}
};
const Json& member(const Json& j,const std::string& k){const auto& o=std::get<Json::Object>(j.value);auto i=o.find(k);need(i!=o.end(),"JSON key "+k);return i->second;}
void json_ints(const Json& j,std::initializer_list<std::int64_t> want){
 const auto& a=std::get<Json::Array>(j.value);need(a.size()==want.size(),"JSON array length");std::size_t i=0;for(auto x:want)need(std::get<std::int64_t>(a[i++].value)==x,"JSON int");
}
void json_texts(const Json& j,std::initializer_list<std::string_view> want){
 const auto& a=std::get<Json::Array>(j.value);need(a.size()==want.size(),"JSON texts length");std::size_t i=0;for(auto x:want)need(std::get<std::string>(a[i++].value)==x,"JSON text");
}
void ipc_cases(const fs::path& d){
 auto f=d/"sounio_arrow_ipc.arrow",s=d/"sounio_arrow_ipc.arrows";
 framing(f,true,true);passed("ipc.file.structural");framing(s,false,true);passed("ipc.stream.structural");
 basic(file(f));passed("ipc.file.logical");basic(stream(s));passed("ipc.stream.logical");
 auto mf=d/"sounio_arrow_ipc_multi.arrow",ms=d/"sounio_arrow_ipc_multi.arrows";
 framing(mf,true,false);framing(ms,false,false);
 for(auto [path,is_file]:{std::pair{mf,true},std::pair{ms,false}}){auto data=is_file?file(path):stream(path);need(data.batches.size()==2,"multi batch count");ints(data.batches[0]->column(0),{1,2},"multi0");ints(data.batches[1]->column(0),{30},"multi1");passed(is_file?"ipc.multi.file":"ipc.multi.stream");}
}
void spec_cases(const fs::path& d){
 auto n=stream(d/"sounio_arrow_nested.arrows");need(n.batches.size()==1,"nested count");need(take(n.schema->metadata()->Get("app:ns"))=="s","schema metadata");
 need(meta(field(n,"col2"),"ARROW:extension:name")=="myorg.word"&&meta(field(n,"col2"),"ARROW:extension:metadata")=="v1","custom extension metadata");
 auto st=std::dynamic_pointer_cast<arrow::StructArray>(col(n.batches[0],"col1"));need(bool(st)&&st->length()==2,"nested struct");
 need(integer(st->field(0),0)==1&&st->field(0)->IsNull(1),"nested a");ints(list(st->field(1),0),{10,11},"nested b0");ints(list(st->field(1),1),{12},"nested b1");
 need(real(st->field(2),0)==1.5&&st->field(2)->IsNull(1),"nested c");auto c2=col(n.batches[0],"col2");need(text(c2,0)=="ab"&&c2->IsNull(1),"nested text");passed("spec.nested");
 auto dict=stream(d/"sounio_arrow_dict.arrows");need(dict.batches.size()==2,"dict batches");texts(dict.batches[0]->column(0),{"A","B","C","B"},"dict0");texts(dict.batches[1]->column(0),{"D","C","E","A"},"dict1");passed("spec.dictionary");
 for(auto codec:{"lz4","zstd"}){auto s=stream(d/(std::string("sounio_arrow_")+codec+".arrows"));need(s.batches.size()==1,"codec batches");texts(col(s.batches[0],"v"),{"hi","abcdefghijklm"},"view");ints(col(s.batches[0],"id"),{1,2},"codec id");ints(col(s.batches[0],"ts"),{7,8},"timestamp");auto t=std::dynamic_pointer_cast<arrow::TimestampType>(field(s,"ts")->type());need(bool(t)&&t->unit()==arrow::TimeUnit::NANO&&t->timezone()=="UTC","timestamp type");passed(std::string("spec.")+codec);}
 auto ree=stream(d/"sounio_arrow_ree.arrows");auto rt=std::dynamic_pointer_cast<arrow::RunEndEncodedType>(field(ree,"r")->type());need(bool(rt)&&rt->run_end_type()->id()==arrow::Type::INT32&&rt->value_type()->id()==arrow::Type::INT64,"REE type");ints(col(ree.batches[0],"r"),{7,7,9,9},"REE");passed("spec.ree");
 auto sc=stream(d/"sounio_arrow_scalars.arrows");need(sc.batches.size()==1,"scalar batches");auto b=sc.batches[0];
 need(field(sc,"d32")->type()->id()==arrow::Type::DATE32&&field(sc,"d64")->type()->id()==arrow::Type::DATE64,"date types");
 auto d32=col(b,"d32");need(integer(d32,0)==19000&&d32->IsNull(1),"date32");ints(col(b,"d64"),{432000000,-777600000},"date64");
 auto t32=std::dynamic_pointer_cast<arrow::Time32Type>(field(sc,"t32")->type());auto t64=std::dynamic_pointer_cast<arrow::Time64Type>(field(sc,"t64")->type());need(bool(t32)&&t32->unit()==arrow::TimeUnit::MILLI&&bool(t64)&&t64->unit()==arrow::TimeUnit::NANO,"time types");ints(col(b,"t32"),{1000,86399999},"time32");ints(col(b,"t64"),{7,8},"time64");need(field(sc,"iym")->type()->id()==arrow::Type::INTERVAL_MONTHS,"month interval");passed("spec.scalar.types");
 auto dec64=std::dynamic_pointer_cast<arrow::Decimal64Type>(field(sc,"dec64")->type());need(bool(dec64)&&dec64->precision()==10&&dec64->scale()==2,"decimal64 type");auto da64=std::dynamic_pointer_cast<arrow::Decimal64Array>(col(b,"dec64"));need(bool(da64)&&da64->FormatValue(0)=="-0.42"&&da64->FormatValue(1)=="0.42","decimal64 values");passed("spec.scalar.dec64");
 auto dec128=std::dynamic_pointer_cast<arrow::Decimal128Type>(field(sc,"dec128")->type());need(bool(dec128)&&dec128->precision()==38&&dec128->scale()==4,"decimal128 type");auto da128=std::dynamic_pointer_cast<arrow::Decimal128Array>(col(b,"dec128"));need(bool(da128)&&da128->FormatValue(0)=="6432940189538350054795618879031647.8768"&&da128->FormatValue(1)=="6405838452101818886974504219670769.8022","decimal128 values");passed("spec.scalar.dec128");
 need(real(col(b,"f32"),0)==-1.0&&real(col(b,"f32"),1)==1.0,"float32 values");passed("spec.scalar.float");
 auto more=stream(d/"sounio_arrow_more.arrows");b=more.batches[0];texts(col(b,"fb"),{"xyz"},"fixed binary");auto mt=std::dynamic_pointer_cast<arrow::MapType>(field(more,"m")->type());need(bool(mt)&&mt->keys_sorted()&&mt->key_type()->id()==arrow::Type::INT32&&mt->item_type()->id()==arrow::Type::INT64,"map type");
 auto ma=std::dynamic_pointer_cast<arrow::StructArray>(list(col(b,"m"),0));need(bool(ma),"map entries");ints(ma->field(0),{10,20},"map keys");ints(ma->field(1),{100,200},"map values");texts(col(b,"lu"),{"pq"},"large utf8");ints(list(col(b,"lv"),0),{5,6},"list view");passed("spec.more");
 auto un=stream(d/"sounio_arrow_unions.arrows");b=un.batches[0];need(field(un,"su")->type()->id()==arrow::Type::SPARSE_UNION&&field(un,"du")->type()->id()==arrow::Type::DENSE_UNION,"union types");ints(col(b,"su"),{4,9},"sparse union");ints(col(b,"du"),{4,9},"dense union");
 auto lt=std::dynamic_pointer_cast<arrow::LargeListType>(field(un,"ll")->type());need(bool(lt)&&lt->value_type()->id()==arrow::Type::INT16&&!lt->value_field()->nullable(),"large list type");ints(list(col(b,"ll"),0),{99},"large list0");ints(list(col(b,"ll"),1),{-5},"large list1");passed("spec.unions");
}
void ext_cases(const fs::path& d){
 auto s=stream(d/"sounio_arrow_ext.arrows");need(s.batches.size()==1,"ext batch");ext(s,"id","arrow.uuid");ext(s,"j","arrow.json");ext(s,"ok","arrow.bool8");auto op=ext(s,"g","arrow.opaque");
 auto b=s.batches[0];need(integer(col(b,"ok"),0)==0&&integer(col(b,"ok"),1)!=0,"bool8 semantics");texts(col(b,"j"),{"{\"a\":1}","null"},"json values");
 auto id=text(col(b,"id"),0);need(id=="0123456789abcdef","uuid bytes");std::string uuid;const char* digits="0123456789abcdef";for(std::size_t i=0;i<id.size();++i){if(i==4||i==6||i==8||i==10)uuid+='-';const auto x=static_cast<unsigned char>(id[i]);uuid+=digits[x>>4];uuid+=digits[x&15];}need(uuid=="30313233-3435-3637-3839-616263646566","uuid semantic format");
 auto oj=JsonReader(op->Serialize()).parse();need(std::get<std::string>(member(oj,"type_name").value)=="geometry"&&std::get<std::string>(member(oj,"vendor_name").value)=="PostGIS","opaque params");passed("ext.simple");
 auto tensor=stream(d/"sounio_arrow_ext_tensor.arrows");auto et=ext(tensor,"t","arrow.fixed_shape_tensor");auto tj=JsonReader(et->Serialize()).parse();json_ints(member(tj,"shape"),{2,3});json_ints(member(tj,"permutation"),{1,0});json_texts(member(tj,"dim_names"),{"a","b"});ints(list(storage(col(tensor.batches[0],"t")),0),{1,2,3,4,5,6},"tensor values");passed("ext.tensor");
 auto escaped=stream(d/"sounio_arrow_ext_escaped.arrows");auto ej=JsonReader(ext(escaped,"t","arrow.fixed_shape_tensor")->Serialize()).parse();json_texts(member(ej,"dim_names"),{"a\"b\n","c\\d\t"});passed("ext.escaped");
 auto structures=stream(d/"sounio_arrow_ext_struct.arrows");need(meta(field(structures,"p"),"ARROW:extension:name")=="arrow.parquet.variant"&&meta(field(structures,"p"),"ARROW:extension:metadata").empty(),"variant metadata");need(meta(field(structures,"when"),"ARROW:extension:name")=="arrow.timestamp_with_offset","timestamp-offset metadata");passed("ext.struct");
}
std::uint64_t varint(const Bytes& b,std::size_t& p,bool value){
 std::uint64_t v=0;const unsigned count=value?10:5;
 for(unsigned i=0;i<count;++i){need(p<b.size(),"bad varint");auto c=b[p++];if(value&&i==9)need(c<=1,"bad varint");v|=std::uint64_t(c&127)<<(i*7);if(c<128)return v;}throw Bad("bad varint");
}
struct Pb {unsigned field;std::variant<std::uint64_t,Bytes> value;};
std::vector<Pb> protobuf(const Bytes& b){
 std::vector<Pb> fields;std::size_t p=0;
 while(p<b.size()){auto tag=varint(b,p,false);auto field=tag>>3;auto wire=tag&7;need(field>=1&&field<=536870911,"protobuf field");
 if(wire==0)fields.push_back({static_cast<unsigned>(field),varint(b,p,true)});
 else if(wire==2){auto n=varint(b,p,false);need(n<=b.size()-p,"protobuf length past end");fields.push_back({static_cast<unsigned>(field),Bytes(b.begin()+static_cast<std::ptrdiff_t>(p),b.begin()+static_cast<std::ptrdiff_t>(p+n))});p+=n;}
 else if(wire==1||wire==5){const std::size_t n=wire==1?8:4;need(n<=b.size()-p,"protobuf length past end");p+=n;}else throw Bad("protobuf wire");
 }return fields;
}
std::vector<Bytes> frames(const Bytes& b){
 std::vector<Bytes> out;std::size_t p=0;while(p<b.size()){need(b.size()-p>=5,"truncated frame");need(b[p]==0,"compressed flag");auto n=u32(b,p+1,true);p+=5;need(n<=b.size()-p,"frame length past end");out.emplace_back(b.begin()+static_cast<std::ptrdiff_t>(p),b.begin()+static_cast<std::ptrdiff_t>(p+n));p+=n;}return out;
}
struct Parts {std::optional<Bytes> desc,header,meta,body;};
Parts parts(const Bytes& b){Parts x;for(auto f:protobuf(b))if(auto v=std::get_if<Bytes>(&f.value)){if(f.field==1)x.desc=*v;else if(f.field==2)x.header=*v;else if(f.field==3)x.meta=*v;else if(f.field==1000)x.body=*v;}return x;}
Bytes to_ipc(const std::vector<Parts>& p){
 Bytes b;for(const auto& x:p){if(!x.header||x.header->empty()){need(!x.body||x.body->empty(),"empty header with a body");continue;}const Bytes empty;const auto& body=x.body?*x.body:empty;need(body.size()%8==0,"body not padded");auto pad=(8-x.header->size()%8)%8;le32(b,0xffffffffU);le32(b,static_cast<std::uint32_t>(x.header->size()+pad));b.insert(b.end(),x.header->begin(),x.header->end());b.insert(b.end(),pad,0);b.insert(b.end(),body.begin(),body.end());}if(!b.empty()){le32(b,0xffffffffU);le32(b,0);}return b;
}
template<class F>void rejects(F f,std::string_view why){try{f();}catch(const Bad& e){need(std::string_view(e.what()).find(why)!=std::string_view::npos,"wrong rejection: "+std::string(e.what()));return;}throw Bad("invalid input accepted");}
void flight_self(){
 for(auto s:{"1208616263","09616263","0d616263"}){rejects([&]{protobuf(hex(s));},"protobuf length past end");}passed("flight.self.truncated");
 std::size_t p=0;varint(hex("808080808001"),p,true);need(p==6,"six-byte varint");p=0;rejects([&]{varint(hex("808080808001"),p,false);},"bad varint");p=0;rejects([&]{varint(hex("80808080808080808002"),p,true);},"bad varint");p=0;varint(hex("80808080808080808001"),p,true);need(p==10,"ten-byte varint");passed("flight.self.varints");
 for(auto s:{"0200","828080801000"}){rejects([&]{protobuf(hex(s));},"protobuf field");}protobuf(hex("faffffff0f00"));passed("flight.self.fieldlimits");
 need(to_ipc({Parts{std::nullopt,Bytes{},std::nullopt,std::nullopt}}).empty()&&to_ipc({Parts{}}).empty(),"empty header emitted EOS");passed("flight.self.emptyheader");
}
void flight_cases(const fs::path& d){
 flight_self();auto raw=frames(bytes(d/"sounio_flight_do_put.bin"));need(raw.size()==3,"DoPut frames");std::vector<Parts> put;for(auto& x:raw)put.push_back(parts(x));
 need(put[0].desc==hex("08011a04646174611a026964")&&!put[1].desc&&!put[2].desc&&!put[0].body,"DoPut descriptor/schema");auto pi=to_ipc(put);need(pi.size()>=4&&u32(pi,0)==0xffffffffU,"Flight IPC continuation");passed("flight.doput");
 auto results=frames(bytes(d/"sounio_flight_put_result.bin"));need(results==std::vector<Bytes>{hex("0a026f6b"),Bytes{}},"PutResult values");passed("flight.results");
 auto cf=frames(bytes(d/"sounio_flight_exchange_client.bin")),sf=frames(bytes(d/"sounio_flight_exchange_server.bin"));need(cf.size()==1&&sf.size()==2,"exchange frames");std::vector<Parts> client{parts(cf[0])},server{parts(sf[0]),parts(sf[1])};
 need(client[0].desc==hex("080212057175657279")&&!server[0].desc&&!server[1].desc,"exchange descriptors");passed("flight.exchange");
 auto dp=take(arrow::flight::FlightDescriptor::Deserialize(std::string(put[0].desc->begin(),put[0].desc->end())));auto dc=take(arrow::flight::FlightDescriptor::Deserialize(std::string(client[0].desc->begin(),client[0].desc->end())));
 need(dp.type==arrow::flight::FlightDescriptor::PATH&&dp.path==std::vector<std::string>{"data","id"},"official Flight path");need(dc.type==arrow::flight::FlightDescriptor::CMD&&dc.cmd=="query","official Flight command");passed("flight.descriptors");
 auto ps=stream_blob(pi),cs=stream_blob(to_ipc(client)),ss=stream_blob(to_ipc(server));need(ps.batches.size()==2&&cs.batches.empty()&&ss.batches.size()==1,"Flight IPC batch counts");ints(ps.batches[0]->column(0),{1,2},"DoPut0");ints(ps.batches[1]->column(0),{30},"DoPut1");ints(ss.batches[0]->column(0),{7},"exchange reply");passed("flight.ipc");
}
void version(){
 need(arrow::GetBuildInfo().version_string=="26.0.0","runtime Arrow version");
 for(auto name:{"arrow.uuid","arrow.json","arrow.bool8","arrow.opaque","arrow.fixed_shape_tensor"})need(bool(arrow::GetExtensionType(name)),"Arrow canonical extension unavailable: "+std::string(name));
 std::cout<<"ARROW_CPP_VERSION 26.0.0\n";
}
void mutant(const fs::path& source,const fs::path& output){
 auto s=file(source);need(s.batches.size()==1,"mutant source");auto b=s.batches[0];arrow::Int64Builder ids;okay(ids.AppendValues({9,2,-3}));auto a=take(ids.Finish());auto cols=b->columns();cols[0]=a;
 auto m=arrow::RecordBatch::Make(s.schema,b->num_rows(),cols);okay(m->ValidateFull());auto out=take(arrow::io::FileOutputStream::Open(output.string()));auto writer=take(arrow::ipc::MakeFileWriter(out,s.schema));okay(writer->WriteRecordBatch(*m));okay(writer->Close());okay(out->Close());std::cout<<"ID_MUTANT_WRITTEN\n";
}

void incremental_write(const fs::path& output){
 auto schema=arrow::schema({arrow::field("id",arrow::int64())});
 auto out=take(arrow::io::FileOutputStream::Open(output.string()));
 auto writer=take(arrow::ipc::MakeStreamWriter(out,schema));
 for(std::int64_t batch=0;batch<100;++batch){
  arrow::Int64Builder builder;for(std::int64_t row=0;row<16;++row)okay(builder.Append(batch*16+row));
  auto array=take(builder.Finish());auto record=arrow::RecordBatch::Make(schema,16,{array});
  okay(record->ValidateFull());okay(writer->WriteRecordBatch(*record));
 }
 okay(writer->Close());okay(out->Close());
 auto original=bytes(output);need(original.size()>8192,"incremental producer did not exceed buffer");
 for(bool trailing:{false,true}){
  auto changed=original;if(trailing)changed.push_back(0);else changed.pop_back();
  std::ofstream f(output.string()+(trailing?".trailing":".truncated"),std::ios::binary);
  need(bool(f),"mutant open");f.write(reinterpret_cast<const char*>(changed.data()),static_cast<std::streamsize>(changed.size()));need(bool(f),"mutant write");
 }
 std::cout<<"ARROW_INCREMENTAL_CPP_WRITTEN\n";
}
void incremental_check(const fs::path& input){
 need(bytes(input).size()>8192,"incremental output size");auto s=stream(input);
 need(s.schema->num_fields()==1&&s.schema->field(0)->name()=="id"&&s.schema->field(0)->type()->Equals(arrow::int64()),"incremental schema");
 need(s.batches.size()==100,"incremental batch count");
 for(std::int64_t b=0;b<100;++b){
  need(s.batches[b]->num_rows()==16,"incremental rows");
  for(std::int64_t r=0;r<16;++r)need(integer(s.batches[b]->column(0),r)==b*16+r,"incremental value");
 }
 std::cout<<"ARROW_INCREMENTAL_CPP_OK\n";
}

void incremental_dictionary_check(const fs::path& input){
 auto s=stream(input);need(s.batches.size()==2,"incremental dictionary batches");
 texts(s.batches[0]->column(0),{"A","B","C","B"},"incremental dictionary0");
 texts(s.batches[1]->column(0),{"D","C","E","A"},"incremental dictionary1");
 std::cout<<"ARROW_INCREMENTAL_DICTIONARY_CPP_OK\n";
}
int main(int argc,char** argv){
 try{version();if(argc==2&&std::string_view(argv[1])=="--version")return 0;
  if(argc==3&&std::string_view(argv[1])=="--check-incremental-dictionary"){incremental_dictionary_check(argv[2]);return 0;}
  if(argc==3&&std::string_view(argv[1])=="--write-incremental"){incremental_write(argv[2]);return 0;}
  if(argc==3&&std::string_view(argv[1])=="--check-incremental"){incremental_check(argv[2]);return 0;}
  if(argc==4&&std::string_view(argv[1])=="--write-id-mutant"){mutant(argv[2],argv[3]);return 0;}
  if(argc==3&&std::string_view(argv[1])=="--check-basic-logical"){basic(file(argv[2]));std::cout<<"BASIC_LOGICAL_OK\n";return 0;}
  need(argc==3&&std::string_view(argv[1])=="--check","usage: arrow_cpp_interop --check <fixture-dir>");fs::path d=argv[2];ipc_cases(d);spec_cases(d);ext_cases(d);flight_cases(d);std::cout<<"ARROW_CPP_INTEROP_OK\n";return 0;
 }catch(const std::exception& e){std::cerr<<"ARROW_CPP_FAIL "<<e.what()<<"\n";return 1;}
}


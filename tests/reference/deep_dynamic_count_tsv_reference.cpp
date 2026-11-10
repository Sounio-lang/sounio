// Exact integer-only count TSV reference. C++23 from_chars line parser is
// independent of production incremental state. Stdout emits Sounio fixture.
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
static constexpr std::int64_t mod=1000000007;
struct Parsed{int status=0;std::int64_t position=0,rows=0,hash=0;};
static std::int64_t add(std::int64_t h,std::int64_t v){return (h*131+v%mod)%mod;}
static Parsed parse(const std::vector<std::int64_t>&b,std::size_t samples,std::int64_t limit){
 Parsed out;std::size_t begin=0;
 auto fail=[&](int status,std::size_t pos){out.status=status;out.position=static_cast<std::int64_t>(pos);return out;};
 while(begin<b.size()){
  std::size_t end=begin;while(end<b.size()&&b[end]!=10&&b[end]!=13)++end;
  std::vector<std::int64_t> row;std::size_t start=begin;
  for(;;){
   std::size_t sep=start;while(sep<end&&b[sep]!=9)++sep;
   std::string token;std::size_t bad=sep;
   for(std::size_t j=start;j<sep;++j){if(b[j]<48||b[j]>57){bad=j;break;}token.push_back(static_cast<char>(b[j]));}
   std::uint64_t value=0;
   if(!token.empty()){
    auto r=std::from_chars(token.data(),token.data()+token.size(),value);
    if(r.ec!=std::errc{}||value>static_cast<std::uint64_t>(limit)){
     for(std::size_t k=1;k<=token.size();++k){
      std::uint64_t prefix=0;auto p=std::from_chars(token.data(),token.data()+k,prefix);
      if(p.ec!=std::errc{}||prefix>static_cast<std::uint64_t>(limit))return fail(5,start+k-1);
     }
     throw std::logic_error("prefix localization");
    }
   }
   if(bad!=sep)return fail(3,bad);
   if(sep<end&&row.size()>=samples-1)return fail(4,sep);
   if(token.empty())return fail(3,sep);
   row.push_back(static_cast<std::int64_t>(value));
   if(sep==end)break;
   start=sep+1;
  }
  if(row.size()!=samples)return fail(4,end);
  std::size_t next=end;
  if(end<b.size()){
   if(b[end]==13){
    if(end+1>=b.size())return fail(3,b.size());
    if(b[end+1]!=10)return fail(3,end+1);
    next=end+2;
   }else next=end+1;
  }
  for(auto value:row)out.hash=add(out.hash,value);
  ++out.rows;begin=next;
 }
 out.position=static_cast<std::int64_t>(b.size());return out;
}
static std::vector<std::int64_t> bytes(const std::string&s){
 std::vector<std::int64_t>b;for(unsigned char c:s)b.push_back(c);return b;
}
int main(){
 try{
 std::cout<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_DYNAMIC_COUNT_TSV_REFERENCE_PASS
use collections::integer_buffer::{IntegerBuffer,integer_buffer_new,integer_buffer_status,integer_buffer_len,integer_buffer_at,integer_buffer_set,integer_buffer_close}
use genomics::io::dynamic_count_tsv::{DynamicCountTsv,DynamicCountTsvFeed,dynamic_count_tsv_new,dynamic_count_tsv_info,dynamic_count_tsv_feed_slot,dynamic_count_tsv_feed_into,dynamic_count_tsv_ack_row,dynamic_count_tsv_finish_into}
struct Checksum {value:i64}
fn hash_row(row:&IntegerBuffer,hash:&!Checksum) with Mut,Panic {
 var j:i64=0
 while j<integer_buffer_len(row) {
  let value=integer_buffer_at(row,j)
  hash.value=(hash.value*131+value%1000000007)%1000000007
  j=j+1
 }
}
fn piece(state:&!DynamicCountTsv,row:&!IntegerBuffer,b:&IntegerBuffer,start:i64,length:i64,hash:&!Checksum,slot:&!DynamicCountTsvFeed)->i64 with Mut,Div,Panic {
 var offset:i64=0
 while offset<length {
  let status=dynamic_count_tsv_feed_into(state,row,b,start+offset,length-offset,slot)
  if status!=0 {return status}
  if slot.row_ready {
   hash_row(row,hash)
   assert(dynamic_count_tsv_ack_row(state,row))
  } else {if slot.consumed==0 {return 6}}
  offset=offset+slot.consumed
 }
 0
}
fn small(b:&IntegerBuffer,samples:i64,limit:i64,status:i64,position:i64,rows:i64,expected_hash:i64) with Alloc,Mut,Div,Panic {
 var row=integer_buffer_new(samples,samples*8)
 assert(integer_buffer_status(&row)==0)
 let n=integer_buffer_len(b)
 var slot=dynamic_count_tsv_feed_slot()
 var split:i64=0
 // Last iteration is the independent every-byte chunking path.
 while split<=n+1 {
  var state=dynamic_count_tsv_new(samples,limit)
  var hash=Checksum{value:0}
  var actual:i64=0
  if split<=n {
   actual=piece(&!state,&!row,b,0,split,&!hash,&!slot)
   if actual==0 {actual=piece(&!state,&!row,b,split,n-split,&!hash,&!slot)}
  } else {
   var k:i64=0
   while k<n {
    actual=piece(&!state,&!row,b,k,1,&!hash,&!slot)
    if actual!=0 {break}
    k=k+1
   }
  }
  if actual==0 {
   actual=dynamic_count_tsv_finish_into(&!state,&!row,&!slot)
   if slot.row_ready {
    hash_row(&row,&!hash)
    assert(dynamic_count_tsv_ack_row(&!state,&!row))
   }
  }
  let info=dynamic_count_tsv_info(&state)
  assert(actual==status && info.status==status && info.position==position)
  assert(info.rows==rows && hash.value==expected_hash)
  if status!=0 {assert(info.error_position==position)}
  split=split+1
 }
 integer_buffer_close(row)
}
struct Producer {row:i64,column:i64,value:i64,divisor:i64,stage:i64,done:bool}
fn fill(p:&!Producer,b:&!IntegerBuffer)->i64 with Mut,Div {
 var used:i64=0
 while used<integer_buffer_len(b) {
  if p.done {break}
  if p.stage==0 {
   p.value=(p.row*17+p.column*13)%100000
   p.divisor=1
   while p.value/p.divisor>=10 {p.divisor=p.divisor*10}
   p.stage=1
  }
  if p.stage==1 {
   let digit=p.value/p.divisor
   integer_buffer_set(b,used,48+digit);used=used+1
   p.value=p.value%p.divisor;p.divisor=p.divisor/10
   if p.divisor==0 {p.stage=2}
  } else if p.stage==2 {
   if p.column<511 {
    integer_buffer_set(b,used,9);used=used+1
    p.column=p.column+1;p.stage=0
   } else {
    if p.row==19999 {p.done=true}
    else {
     if p.row%2==0 {
      integer_buffer_set(b,used,10);used=used+1
      p.row=p.row+1;p.column=0;p.stage=0
     } else {
      integer_buffer_set(b,used,13);used=used+1;p.stage=3
     }
    }
   }
  } else if p.stage==3 {
   integer_buffer_set(b,used,10);used=used+1
   p.row=p.row+1;p.column=0;p.stage=0
  }
 }
 used
}
fn large(expected_bytes:i64,expected_hash:i64) with IO,Alloc,Mut,Div,Panic {
 var row=integer_buffer_new(512,4096)
 var chunk=integer_buffer_new(4096,32768)
 assert(integer_buffer_status(&row)==0 && integer_buffer_status(&chunk)==0)
 var producer=Producer{row:0,column:0,value:0,divisor:1,stage:0,done:false}
 var reader=dynamic_count_tsv_new(512,9223372036854775807)
 var hash=Checksum{value:0}
 var slot=dynamic_count_tsv_feed_slot()
 while !producer.done {
  let n=fill(&!producer,&!chunk)
  let status=piece(&!reader,&!row,&chunk,0,n,&!hash,&!slot)
  assert(status==0)
 }
 let final=dynamic_count_tsv_finish_into(&!reader,&!row,&!slot)
 assert(final==0 && slot.row_ready && slot.finished)
 hash_row(&row,&!hash)
 assert(dynamic_count_tsv_ack_row(&!reader,&!row))
 let info=dynamic_count_tsv_info(&reader)
 assert(info.rows==20000 && info.position==expected_bytes && info.finished)
 assert(hash.value==expected_hash)
 integer_buffer_close(row);integer_buffer_close(chunk)
}
fn single_byte_stream(expected_bytes:i64,expected_rows:i64) with IO,Alloc,Mut,Div,Panic {
 var row=integer_buffer_new(1,8)
 var byte=integer_buffer_new(1,8)
 assert(integer_buffer_status(&row)==0 && integer_buffer_status(&byte)==0)
 var state=dynamic_count_tsv_new(1,0)
 var slot=dynamic_count_tsv_feed_slot()
 var i:i64=0
 var completed:i64=0
 while i<expected_bytes {
  integer_buffer_set(&!byte,0,if i%2==0 {48} else {10})
  let status=dynamic_count_tsv_feed_into(&!state,&!row,&byte,0,1,&!slot)
  assert(status==0 && slot.consumed==1)
  if slot.row_ready {
   assert(integer_buffer_at(&row,0)==0)
   assert(dynamic_count_tsv_ack_row(&!state,&!row))
   completed=completed+1
  }
  i=i+1
 }
 let final=dynamic_count_tsv_finish_into(&!state,&!row,&!slot)
 assert(final==0 && slot.finished && !slot.row_ready && slot.position==expected_bytes)
 let info=dynamic_count_tsv_info(&state)
 assert(info.rows==expected_rows && completed==expected_rows && info.position==expected_bytes)
 integer_buffer_close(row);integer_buffer_close(byte)
}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 struct Case{std::size_t samples;std::int64_t max;std::vector<std::int64_t> data;};
 const auto max=std::numeric_limits<std::int64_t>::max();
 std::vector<Case>cases={
 {2,max,bytes("1\t2\r\n3\t4")},{2,max,bytes("0\t00\n")},
 {1,max,bytes("9223372036854775807\n")},{1,max,bytes("9223372036854775808\n")},
 {1,9007199254740992LL,bytes("9007199254740992\n")},
 {1,9007199254740992LL,bytes("9007199254740993\n")},
 {2,max,bytes("1\t\n")},{2,max,bytes("\t2\n")},{2,max,bytes("1\n")},
 {2,max,bytes("1\t2\t3\n")},{2,max,bytes("-1\t2\n")},{2,max,bytes("+1\t2\n")},
 {2,max,bytes("1 2\t3\n")},{2,max,bytes("1\t2\rX")},{2,max,bytes("1\t2\r")},
 {2,max,bytes("\n")},{2,max,bytes("")},{1,0,bytes("0")},{1,0,bytes("1")},
 {1,max,bytes("1\n2\n")},{1,max,{128}},{1,max,{-1}},{1,max,{256}}
 };
 for(const auto&c:cases){
  const auto result=parse(c.data,c.samples,c.max);
  std::cout<<" {\n var b=integer_buffer_new("<<c.data.size()<<","<<c.data.size()*8<<")\n";
  for(std::size_t i=0;i<c.data.size();++i)std::cout<<" integer_buffer_set(&!b,"<<i<<","<<c.data[i]<<")\n";
  std::cout<<" small(&b,"<<c.samples<<","<<c.max<<","<<result.status<<","<<result.position<<","<<result.rows<<","<<result.hash<<")\n integer_buffer_close(b)\n }\n";
 }
 std::int64_t count_bytes=0,hash=0,formula_hash=0;
 for(std::int64_t r=0;r<20000;++r){
  std::string line;
  for(std::int64_t c=0;c<512;++c){
   auto value=(r*17+c*13)%100000;
   formula_hash=add(formula_hash,value);
   line+=std::to_string(value);if(c!=511)line+='\t';
  }
  if(r!=19999)line+=r%2==0?"\n":"\r\n";
  auto result=parse(bytes(line),512,max);
  if(result.status!=0||result.rows!=1)throw std::runtime_error("independent generated row parse");
  // Reparse each field through from_chars into a cumulative checksum,
  // instead of attempting to combine modular row hashes with wrong lengths.
  std::size_t pos=0;
  for(int c=0;c<512;++c){
   auto end=line.find_first_of("\t\r\n",pos);if(end==std::string::npos)end=line.size();
   std::int64_t v=0;auto converted=std::from_chars(line.data()+pos,line.data()+end,v);
   if(converted.ec!=std::errc{}||converted.ptr!=line.data()+end)throw std::runtime_error("field parse");
   hash=add(hash,v);pos=end+1;
  }
  count_bytes+=static_cast<std::int64_t>(line.size());
 }
 if(hash!=formula_hash)throw std::runtime_error("formula/parser hash mismatch");
 std::int64_t single_bytes=0,single_rows=0;
 const auto pattern=bytes("0\n");
 for(int i=0;i<500000;++i){
  const auto result=parse(pattern,1,0);
  if(result.status!=0||result.rows!=1)throw std::runtime_error("single-byte expected stream");
  single_bytes+=result.position;single_rows+=result.rows;
 }
 std::cout<<" large("<<count_bytes<<","<<hash<<")\n single_byte_stream("<<single_bytes<<","<<single_rows<<")\n println(\"STDLIB_DEEP_DYNAMIC_COUNT_TSV_REFERENCE_PASS\")\n 0\n}\n";
 return std::cout.good()?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

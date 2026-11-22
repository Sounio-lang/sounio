// C++23 exact count-TSV reference and Sounio fixture emitter.
// Uses std::from_chars on whole decimal tokens/prefixes, independent of the
// production incremental multiply/add state machine. No floating arithmetic.
// g++ -std=c++23 -O2 -Wall -Wextra -pedantic count_tsv_reference.cpp -o count_tsv_reference
// ./count_tsv_reference --fixture > genomics_count_tsv.sio
#include <array>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using I=std::int64_t;using Row=std::array<I,8>;
struct Case{std::string name,text;int samples;};
struct Parsed{int status=0;std::size_t position=0;std::vector<Row> rows;};
Parsed parse(const Case&c){
 Parsed out;
 if(c.samples<1||c.samples>8){out.status=1;return out;}
 Row row{};int column=0;std::string token;
 auto error=[&](int status,std::size_t pos){out.status=status;out.position=pos;return out;};
 auto field=[&](){I value=0;if(token.empty())return 3;auto r=std::from_chars(token.data(),token.data()+token.size(),value);if(r.ec!=std::errc()||r.ptr!=token.data()+token.size())return 5;if(column>=c.samples)return 4;row[column++]=value;token.clear();return 0;};
 std::size_t pos=0;
 while(pos<c.text.size()){
  unsigned char ch=static_cast<unsigned char>(c.text[pos]);
  if(ch>='0'&&ch<='9'){
   token.push_back(static_cast<char>(ch));
   I value=0;auto r=std::from_chars(token.data(),token.data()+token.size(),value);
   if(r.ec==std::errc::result_out_of_range)return error(5,pos);
   ++pos;continue;
  }
  if(ch=='\t'){
   if(column>=c.samples-1)return error(4,pos);
   int status=field();if(status)return error(status,pos);
   ++pos;continue;
  }
  if(ch=='\n'||ch=='\r'){
   int status=field();if(status)return error(status,pos);
   if(column!=c.samples)return error(4,pos);
   ++pos;
   if(ch=='\r'){
    if(pos==c.text.size()||c.text[pos]!='\n')return error(3,pos);
    ++pos;
   }
   out.rows.push_back(row);row={};column=0;continue;
  }
  return error(3,pos);
 }
 if(!token.empty()||column!=0){
  int status=field();if(status)return error(status,pos);
  if(column!=c.samples)return error(4,pos);
  out.rows.push_back(row);
 }
 out.position=pos;return out;
}
std::vector<Case> cases(){
 return{
 {"visual_mixed_newlines","0\t2\t4\r\n3\t0\t0\n0\t6\t0\r\n5\t7\t8",3},
 {"empty","",3},{"lf","1\t2\n",2},{"crlf","1\t2\r\n",2},
 {"eof_row","1\t2",2},{"maximum","9223372036854775807\n",1},
 {"maximum_plus_one","9223372036854775808\n",1},
 {"oversized_decimal","999999999999999999999999999999\n",1},
 {"leading_zeroes","000000000000000000000001\t000\n",2},
 {"short_row","1\n",2},{"extra_column","1\t2\t3\n",2},
 {"empty_first","\t1\n",2},{"empty_last","1\t\n",2},
 {"trailing_tab_eof","1\t",2},{"blank_row","\n",1},
 {"lone_cr","1\r",1},{"wrong_after_cr","1\rx",1},
 {"negative","-1\n",1},{"plus","+1\n",1},{"space"," 1\n",1},
 {"decimal","1.0\n",1},{"comma","1,2\n",2},
 {"valid_prefix_bad_suffix","1\t2\n3\tx\n",2},
 {"invalid_samples","1\n",0}
 };
}
struct Qc{int status=0;I rows=0;Row totals{},zeros{};};
void add(Qc&q,const Row&r,int samples){
 if(q.status)return;
 for(int j=0;j<samples;++j)if(r[j]<0){q.status=2;return;}
 // Unsigned sum of two valid nonnegative signed-i64 values fits uint64_t.
 for(int j=0;j<samples;++j)
  if(static_cast<std::uint64_t>(q.totals[j])+static_cast<std::uint64_t>(r[j])>
     static_cast<std::uint64_t>(std::numeric_limits<I>::max())){q.status=5;return;}
 if(q.rows==std::numeric_limits<I>::max()){q.status=5;return;}
 for(int j=0;j<samples;++j){q.totals[j]+=r[j];if(r[j]==0)++q.zeros[j];}
 ++q.rows;
}
const char* helpers=R"SIO(
fn drive(bytes:&[i64;256],length:i64,samples:i64,chunk:i64,expected_status:i64,expected_position:i64,expected:&[i64;2048],expected_rows:i64) with Mut,Div,Panic {
 var state=count_tsv_new(samples)
 var matrix:[i64;2048]=[-7;2048]
 var rows:i64=0
 var position:i64=0
 while position<length && state.status==0 {
  var part:[i64;256]=[0;256]
  var n=length-position
  if n>chunk {n=chunk}
  var i:i64=0
  while i<n {part[i as usize]=bytes[(position+i) as usize] i=i+1}
  let r=count_tsv_feed(&!state,&part,n)
  assert(r.consumed>=0 && r.consumed<=n)
  position=position+r.consumed
  if r.row_ready {
   let appended=count_matrix_append_checked(&!matrix,rows,&state.row,samples)
   assert(appended.status==0)
   rows=appended.rows
   assert(count_tsv_ack_row(&!state))
  } else {assert(r.consumed>0 || r.status!=0)}
 }
 if state.status==0 {
  let eof=count_tsv_finish(&!state)
  if eof.row_ready {
   let appended=count_matrix_append_checked(&!matrix,rows,&state.row,samples)
   assert(appended.status==0)
   rows=appended.rows
   assert(count_tsv_ack_row(&!state))
  }
  let again=count_tsv_finish(&!state)
  if again.status==0 {assert(again.finished)}
 }
 assert(state.status==expected_status)
 assert(state.position==expected_position)
 assert(rows==expected_rows && state.rows==expected_rows)
 if expected_status!=0 {assert(state.error_position==expected_position)}
 var i:i64=0
 while i<rows*8 {assert(matrix[i as usize]==expected[i as usize]) i=i+1}
 while i<2048 {assert(matrix[i as usize]== -7) i=i+1}
}
fn boundaries() with Mut,Div,Panic {
 var state=count_tsv_new(1)
 var bytes:[i64;256]=[0;256]
 bytes[0]=49 bytes[1]=10 bytes[2]=50
 let first=count_tsv_feed(&!state,&bytes,3)
 assert(first.consumed==2 && first.row_ready && !first.finished && state.row[0]==1)
 let paused=count_tsv_feed(&!state,&bytes,3)
 assert(paused.consumed==0 && paused.row_ready && state.position==2)
 let paused_eof=count_tsv_finish(&!state)
 assert(paused_eof.row_ready && !paused_eof.finished)
 assert(count_tsv_ack_row(&!state))
 bytes[0]=50
 let last=count_tsv_feed(&!state,&bytes,1)
 assert(last.consumed==1 && !last.row_ready)
 let eof=count_tsv_finish(&!state)
 assert(eof.row_ready && eof.finished && state.row[0]==2 && state.rows==2)
 assert(count_tsv_ack_row(&!state))
 assert(state.finished)
 let empty=count_tsv_feed(&!state,&bytes,0)
 assert(empty.status==0 && empty.finished)
 let after=count_tsv_feed(&!state,&bytes,1)
 assert(after.status==6 && after.consumed==0)
 assert(count_tsv_finish(&!state).status==6)
 var invalid=count_tsv_new(1)
 assert(count_tsv_feed(&!invalid,&bytes,257).status==2)
 assert(count_tsv_feed(&!invalid,&bytes,1).consumed==0)
 var invalid_byte=count_tsv_new(1)
 bytes[0]=256
 let rejected=count_tsv_feed(&!invalid_byte,&bytes,1)
 assert(rejected.status==3 && rejected.consumed==0 && rejected.error_position==0)
 var split=count_tsv_new(1)
 bytes[0]=49 bytes[1]=13
 assert(count_tsv_feed(&!split,&bytes,2).consumed==2)
 assert(split.pending_cr && !split.ready)
 bytes[0]=10
 let lf=count_tsv_feed(&!split,&bytes,1)
 assert(lf.consumed==1 && lf.row_ready && split.row[0]==1)
}
fn qc_atomic_and_capacity() with Mut,Div,Panic {
 var qc=count_qc_new(3)
 let row:[i64;8]=[9223372036854775807,1,0,-9,-9,-9,-9,-9]
 assert(count_qc_add_row(&!qc,&row)==0)
 let overflow:[i64;8]=[1,0,0,0,0,0,0,0]
 assert(count_qc_add_row(&!qc,&overflow)==5)
 assert(qc.rows==1 && qc.totals[0]==9223372036854775807 && qc.totals[1]==1 && qc.totals[2]==0)
 assert(qc.zeros[0]==0 && qc.zeros[1]==0 && qc.zeros[2]==1)
 assert(count_qc_add_row(&!qc,&row)==5 && qc.rows==1)
 var negative=count_qc_new(3)
 let valid:[i64;8]=[2,3,4,0,0,0,0,0]
 assert(count_qc_add_row(&!negative,&valid)==0)
 let bad:[i64;8]=[8,9,-1,0,0,0,0,0]
 assert(count_qc_add_row(&!negative,&bad)==2)
 assert(negative.rows==1 && negative.totals[0]==2 && negative.totals[1]==3 && negative.totals[2]==4)
 var malformed=count_qc_new(1)
 malformed.zeros[0]=1
 assert(count_qc_add_row(&!malformed,&valid)==6 && malformed.rows==0 && malformed.totals[0]==0)
 var matrix:[i64;2048]=[-7;2048]
 var i:i64=0
 while i<256 {
  var values:[i64;8]=[0;8]
  var j:i64=0
  while j<8 {values[j as usize]=i*10+j j=j+1}
  let a=count_matrix_append_checked(&!matrix,i,&values,8)
  assert(a.status==0 && a.rows==i+1)
  i=i+1
 }
 i=0
 while i<256 {
  var j:i64=0
  while j<8 {assert(matrix[(i*8+j) as usize]==i*10+j) j=j+1}
  i=i+1
 }
 let full=count_matrix_append_checked(&!matrix,256,&valid,8)
 assert(full.status==8 && full.rows==256 && matrix[0]==0 && matrix[2047]==2557)
 var partial:[i64;2048]=[-7;2048]
 let failed=count_matrix_append_checked(&!partial,0,&bad,3)
 assert(failed.status==2 && failed.rows==0 && partial[0]== -7 && partial[7]== -7)
 let inactive=count_matrix_append_checked(&!partial,0,&row,3)
 assert(inactive.status==0 && inactive.rows==1 && partial[0]==9223372036854775807 && partial[1]==1 && partial[2]==0)
 assert(partial[3]==0 && partial[7]==0 && partial[8]== -7)
}
)SIO";
int main(int argc,char**argv){
 bool fixture=argc==2&&std::string(argv[1])=="--fixture";
 if(argc>1&&!fixture)return 2;
 const auto all=cases();
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: GENOMICS_COUNT_TSV_PASS\n// Generated C++23 from_chars exact parser reference.\nuse genomics::io::count_tsv::{count_tsv_new,count_tsv_feed,count_tsv_finish,count_tsv_ack_row,count_qc_new,count_qc_add_row,count_matrix_append_checked}\n"<<helpers;
 for(std::size_t k=0;k<all.size();++k){
  auto c=all[k];auto r=parse(c);
  if(!fixture){std::cout<<"parse\t"<<k<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<r.position<<"\t"<<r.rows.size()<<"\n";continue;}
  std::cout<<"fn case_"<<k<<"() with Mut,Div,Panic {\n var bytes:[i64;256]=[0;256]\n var expected:[i64;2048]=[0;2048]\n";
  for(std::size_t i=0;i<c.text.size();++i)std::cout<<" bytes["<<i<<"]="<<static_cast<unsigned>(static_cast<unsigned char>(c.text[i]))<<"\n";
  for(std::size_t i=0;i<r.rows.size();++i)for(int j=0;j<8;++j)if(r.rows[i][j])std::cout<<" expected["<<i*8+j<<"]="<<r.rows[i][j]<<"\n";
  for(int chunk:{1,2,3,7,256})std::cout<<" drive(&bytes,"<<c.text.size()<<","<<c.samples<<","<<chunk<<","<<r.status<<","<<r.position<<",&expected,"<<r.rows.size()<<")\n";
  std::cout<<"}\n";
 }
 Qc qc;auto visual=parse(all[0]);for(auto row:visual.rows)add(qc,row,3);
 Qc atomic;Row first{std::numeric_limits<I>::max(),1,0};add(atomic,first,3);Row plus{1,0,0};add(atomic,plus,3);
 if(qc.status||qc.rows!=4||atomic.status!=5||atomic.rows!=1)return 3;
 if(!fixture){
  std::cout<<"qc_visual\t"<<qc.rows;for(int j=0;j<3;++j)std::cout<<"\t"<<qc.totals[j]<<"\t"<<qc.zeros[j];std::cout<<"\n";
  std::cout<<"qc_overflow_atomic\t"<<atomic.status<<"\t"<<atomic.rows;for(int j=0;j<3;++j)std::cout<<"\t"<<atomic.totals[j]<<"\t"<<atomic.zeros[j];std::cout<<"\n";
  std::vector<Row> matrix(256);for(std::size_t i=0;i<matrix.size();++i)for(int j=0;j<8;++j)matrix[i][j]=static_cast<I>(i)*10+j;
  std::cout<<"matrix_capacity\t"<<matrix.size()<<"\t"<<matrix[0].size()<<"\t"<<matrix.front().front()<<"\t"<<matrix.back().back()<<"\n";return 0;
 }
 std::cout<<"fn visual_qc() with Mut,Div,Panic {\n var qc=count_qc_new(3)\n";
 for(std::size_t i=0;i<visual.rows.size();++i){auto row=visual.rows[i];std::cout<<" let r"<<i<<":[i64;8]=[";for(int j=0;j<8;++j)std::cout<<(j?",":"")<<row[j];std::cout<<"]\n assert(count_qc_add_row(&!qc,&r"<<i<<")==0)\n";}
 std::cout<<" assert(qc.rows=="<<qc.rows<<")\n";
 for(int j=0;j<3;++j)std::cout<<" assert(qc.totals["<<j<<"]=="<<qc.totals[j]<<" && qc.zeros["<<j<<"]=="<<qc.zeros[j]<<")\n";
 std::cout<<"}\nfn main() with IO,Mut,Div,Panic {\n";
 for(std::size_t k=0;k<all.size();++k)std::cout<<" case_"<<k<<"()\n";
 std::cout<<" boundaries()\n qc_atomic_and_capacity()\n visual_qc()\n println(\"GENOMICS_COUNT_TSV_PASS\")\n 0\n}\n";
}

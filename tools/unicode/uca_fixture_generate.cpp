// C++23 test fixture generator: pinned official UCA ordering + ICU-validated root options.
// This exports expected test rows only. Production Sounio never links ICU.
#define SOUNIO_UCA_REFERENCE_NO_MAIN
#include "../../tests/reference/wave3_uca_reference.cpp"
#include <iomanip>
struct FixtureWriter {
 fs::path root;std::ofstream manifest;Bytes payload;U rows=0,index=0;std::uint64_t total=0;
 explicit FixtureWriter(fs::path p):root(std::move(p)){if(fs::exists(root))fail("output must be new");fs::create_directories(root);manifest.open(root/"MANIFEST.tsv");manifest<<"filename\tkind\trows\tpayload_bytes\n";}
 void flush(){if(!rows)return;std::ostringstream name;name<<"collation-"<<std::setw(6)<<std::setfill('0')<<index++<<".bin";Bytes out={'S','O','U','N','U','T','F','1'};put(out,4);put(out,rows);put(out,narrow(payload.size()));put(out,0x120000);out.insert(out.end(),payload.begin(),payload.end());std::ofstream f(root/name.str(),std::ios::binary);f.write(reinterpret_cast<const char*>(out.data()),out.size());if(!f)fail("write fixture");manifest<<name.str()<<"\t4\t"<<rows<<'\t'<<payload.size()<<'\n';payload.clear();rows=0;}
 void add(U root_mode,U strength,bool shifted,bool backwards,int expected,const Seq&a,const Seq&b){Bytes row;put(row,root_mode);put(row,strength);put(row,shifted);put(row,backwards);put(row,expected+1);put(row,narrow(a.size()));for(U cp:a)put(row,cp);put(row,narrow(b.size()));for(U cp:b)put(row,cp);if(row.size()>1048576)fail("oversized row");if(rows==512||payload.size()+row.size()>1048576)flush();payload.insert(payload.end(),row.begin(),row.end());++rows;++total;}
 void finish(){flush();manifest.flush();if(!manifest||!total)fail("empty/failed fixture manifest");std::ofstream(root/"SCOPE.txt")<<"EXHAUSTIVE_OFFICIAL_UCA_ROOT_CORPUS_AND_FOCUSED_CLDR_ROOT_OPTIONS\nNo locale tailoring acceptance claim.\n";std::cout<<"UCA_FIXTURES_OK rows="<<total<<" chunks="<<index<<"\n";}
};
int main(int argc,char**argv){try{
 if(argc!=7){std::cerr<<"usage: uca_fixture_generate DUCET CLDR_ALLKEYS BLOB NON_IGNORABLE SHIFTED NEW_DIR\n";return 64;}
 auto ducet=read_allkeys(argv[1],false),cldr=read_allkeys(argv[2],true);UcaBlob blob(argv[3]);if(blob.mappings(false)!=ducet.entries||blob.mappings(true)!=cldr.entries)fail("source/binary mismatch");
 RootCollator standard(blob.mappings(false),ducet.implicit),cldr_root(blob.mappings(true),ducet.implicit);FixtureWriter writer(argv[6]);
 for(int file=4;file<6;++file){std::ifstream in(argv[file]);std::string line;Seq previous_input,previous_key;bool have=false,version=false;std::uint64_t count=0;while(std::getline(in,line)){if(line.find("# UCA Version: 18.0.0")!=line.npos)version=true;line=trim(line.substr(0,line.find('#')));if(line.empty())continue;auto input=sequence(line.substr(0,line.find(';')));auto current=standard.key(input,file==5,file==5?4:3);if(have){int expected=order(previous_key,current);if(expected>0)fail("official ordering defect");writer.add(0,file==5?4:3,file==5,false,expected,previous_input,input);}previous_key=std::move(current);previous_input=std::move(input);have=true;++count;}if(!version||count<2)fail("empty/versionless official corpus");std::cout<<"UCA_OFFICIAL_FIXTURE_ROWS source="<<count<<"\n";}
 auto controls=root_control_inputs();for(int strength=1;strength<=5;++strength)for(bool shifted:{false,true})for(bool backwards:{false,true}){std::vector<Seq> keys;for(const auto&s:controls)keys.push_back(cldr_root.key(s,shifted,strength,backwards));for(std::size_t a=0;a<controls.size();++a)for(std::size_t b=0;b<controls.size();++b)writer.add(1,strength,shifted,backwards,order(keys[a],keys[b]),controls[a],controls[b]);}
 // DUCET valid-scalar parity also covers every strength/shifted/backwards option.
 for(int strength=1;strength<=5;++strength)for(bool shifted:{false,true})for(bool backwards:{false,true})for(const auto&input:controls)writer.add(0,strength,shifted,backwards,0,input,input);
 for(int strength=1;strength<=5;++strength)for(bool shifted:{false,true})for(bool backwards:{false,true}){Seq a={0xD800,0xDC00},b={0x10000};writer.add(0,strength,shifted,backwards,order(standard.key(a,shifted,strength,backwards),standard.key(b,shifted,strength,backwards)),a,b);}
 writer.finish();return 0;
}catch(const std::exception&e){std::cerr<<"UCA_FIXTURE_ERROR "<<e.what()<<"\n";return 1;}}


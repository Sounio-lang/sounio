// Complete official CLDR metadata -> observable collator-identity concordance.
// C++23 is the independent implementation reference; this exports no ICU CEs.
#define SOUNIO_CLDR_BUILD_NO_MAIN
#include "cldr_build.cpp"
#include "cldr_locale.hpp"
#include <iomanip>
using Mbytes=std::vector<std::uint8_t>;
void mput(Mbytes&out,std::uint32_t value){for(unsigned i=0;i<4;++i)out.push_back(static_cast<std::uint8_t>(value>>(i*8)));}
void mstring(Mbytes&out,const std::string&value){if(value.size()>255)bad("metadata fixture string extent");mput(out,static_cast<std::uint32_t>(value.size()));out.insert(out.end(),value.begin(),value.end());}
struct Mrow{unsigned kind;std::string input,type,locale,resolved_type;};
int main(int argc,char**argv){try{if(argc!=5)return 64;cldr_input_versions(argv[1],argv[2]);ClocaleRegistry r(argv[1]);std::vector<Mrow> rows;
 auto add=[&](unsigned kind,const std::string&input,const std::string&type){auto id=r.resolve_request(input,type);const auto&target=r.entries.at(id);rows.push_back({kind,input,type,target.locale,target.type});};
 for(const auto&[from,to]:r.language_aliases){(void)to;add(1,from,"");}
 for(const auto&[from,to]:r.script_aliases){(void)to;add(2,"en_"+from,"");}
 for(const auto&[from,to]:r.territory_aliases){(void)to;add(3,"en_"+from,"");}
 for(const auto&[from,to]:r.variant_aliases){(void)to;add(4,"und_"+from,"");}
 for(const auto&[from,to]:r.likely){(void)to;add(5,from,"");}
 for(const auto&[from,to]:r.type_aliases){std::string locale="root";auto canonical=r.canonical_type(to);for(const auto&entry:r.entries)if(entry.type==canonical&&!entry.type.starts_with("private-")&&entry.variant.empty()){locale=entry.locale;break;}add(6,locale,from);}
 for(const auto&entry:r.entries)if(!entry.type.starts_with("private-")&&entry.variant.empty())add(7,entry.locale,entry.type);

 fs::path directory=argv[3];if(!fs::create_directory(directory))bad("metadata output directory must be fresh");
 std::ofstream report(argv[4]);report<<"row\tkind\trequest_locale\trequest_type\tresolved_locale\tresolved_type\n";
 for(std::size_t i=0;i<rows.size();++i){const auto&row=rows[i];report<<i<<'\t'<<row.kind<<'\t'<<row.input<<'\t'<<row.type<<'\t'<<row.locale<<'\t'<<row.resolved_type<<'\n';}
 std::ofstream manifest(directory/"MANIFEST.tsv");manifest<<"filename\tkind\trows\tpayload_bytes\n";
 std::size_t files=0;for(std::size_t first=0;first<rows.size();first+=128){Mbytes payload;auto count=std::min<std::size_t>(128,rows.size()-first);for(std::size_t j=0;j<count;++j){const auto&row=rows[first+j];mput(payload,row.kind);mstring(payload,row.input);mstring(payload,row.type);mstring(payload,row.locale);mstring(payload,row.resolved_type);}
 if(payload.empty()||payload.size()>1048576)bad("metadata fixture payload extent");
 Mbytes output={'S','O','U','N','U','T','F','1'};mput(output,6);mput(output,static_cast<std::uint32_t>(count));mput(output,static_cast<std::uint32_t>(payload.size()));mput(output,0x120000);output.insert(output.end(),payload.begin(),payload.end());
 std::ostringstream name;name<<"locale-meta-"<<std::setfill('0')<<std::setw(6)<<files<<".bin";std::ofstream binary(directory/name.str(),std::ios::binary);binary.write(reinterpret_cast<const char*>(output.data()),static_cast<std::streamsize>(output.size()));if(!binary)bad("metadata fixture write");manifest<<name.str()<<"\t6\t"<<count<<'\t'<<payload.size()<<'\n';++files;}
 std::ofstream scope(directory/"SCOPE.txt");scope<<"COMPLETE_CLDR49_METADATA_CONCORDANCE rows="<<rows.size()<<" files="<<files<<" source=all_alias_likely_type_alias_public_rows no_metadata_rows_skipped Cpp23_identity_oracle_not_ICU_CE_or_full_collation_corpus\n";
 if(!manifest||!report||!scope)bad("metadata manifest write");
 std::cout<<"CLDR_METADATA_FIXTURE_OK files="<<files<<" rows="<<rows.size()<<" language_aliases="<<r.language_aliases.size()<<" script_aliases="<<r.script_aliases.size()<<" territory_aliases="<<r.territory_aliases.size()<<" variant_aliases="<<r.variant_aliases.size()<<" likely="<<r.likely.size()<<" type_aliases="<<r.type_aliases.size()<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"CLDR_METADATA_FIXTURE_ERROR "<<e.what()<<"\n";return 1;}}


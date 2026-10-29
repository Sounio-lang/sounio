#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <cstdint>
int main(int argc,char**argv){
    if(argc!=3){return 2;}
    std::ifstream in(argv[1]);std::map<std::string,std::int64_t> rows;
    std::string key;std::int64_t value=0;
    bool marker=false;while(in>>key){if(key=="STORAGE_CLOSED_AUDIT_CAPTURED"){marker=true;break;}if(!(in>>value)||!rows.emplace(key,value).second){return 3;}}
    if(!marker||in>>key){return 4;}
    std::map<std::string,std::int64_t> expected={
        {"source4352_live",1},{"source4352_count",4352},{"source4353_live",1},{"source4353_count",4353},
        {"source4353_scalar_state",0},{"source4353_nominal_state",2},
        {"same_epoch_field_mutation_state",0},{"same_epoch_field_mutation_equal",1},{"same_epoch_field_mutation_old_new_equal",1},{"same_epoch_ref_mutation_state",1},{"same_epoch_ref_mutation_old_equal",0},
        {"compat_direct_storage",0},{"compat_tuple_storage_self",0},{"compat_tuple_storage_unknown",1},{"compat_array_storage_unknown",1},{"compat_named_inner_storage_int",1},{"compat_tuple_storage_error",1},{"compat_array_storage_error",1},{"compat_named_storage_unknown",1},{"compat_named_storage_error",1},{"compat_named_storage_self",1},
        {"concrete_nominal_extra_inner_state",0},{"tuple_extra_extent_state",0},{"tuple_extra_extent_equal",1},
        {"identity_last_succeeds",1},{"identity_count_at_limit",65536},{"identity_after_limit_succeeds",0},{"identity_count_after_refusal",65536},
        {"identity_partial_published",0},{"identity_partial_count",65536},{"old_context_after_begin",0},{"old_wrapper_new_context",0}};
    if(std::string(argv[2])=="candidate"){expected["source4353_live"]=0;expected["source4353_count"]=4352;expected["source4353_scalar_state"]=2;}else if(std::string(argv[2])!="original"){return 5;}
    if(rows!=expected){for(const auto&[k,v]:expected){auto it=rows.find(k);if(it==rows.end()||it->second!=v){std::cerr<<k<<" expected="<<v<<" observed="<<(it==rows.end()?-999:it->second)<<'\n';}}return 1;}
    std::cout<<"STORAGE_AUDIT_CXX23_CAPTURE_MATCH "<<rows.size()<<" rows; original or isolated candidate captured, public admission absent\n";
}

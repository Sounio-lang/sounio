// Independent VCF 4.3 scalar and genotype-cardinality reference.
// No parser or scheduler implementation is mirrored here.
#include <cassert>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>
static bool integer(std::string_view s) {
 if(s.starts_with("+")) s.remove_prefix(1);
 std::int64_t value=0;
 auto [end,ec]=std::from_chars(s.data(),s.data()+s.size(),value);
 return ec==std::errc{} && end==s.data()+s.size() &&
        value>=-2147483640LL && value<=2147483647LL;
}
static bool floating(const std::string& s) {
 static const std::regex grammar(R"([+-]?([0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?|INF(INITY)?|NAN))",std::regex::icase);
 return s=="." || std::regex_match(s,grammar);
}
// Enumerate unordered genotype tuples, independently of the product formula.
static std::uint64_t genotypes(unsigned alleles,unsigned ploidy,unsigned minimum=0) {
 if(ploidy==0) return 1;
 std::uint64_t result=0;
 for(unsigned allele=minimum;allele<alleles;++allele)
  result+=genotypes(alleles,ploidy-1,allele);
 return result;
}
int main() {
 for(auto value:{"-2147483640","2147483647","+0","-0"}) assert(integer(value));
 for(auto value:{"-2147483641","2147483648","9223372036854775808","1.0","+"}) assert(!integer(value));
 for(auto value:{".",".5","+Inf","-NAN","infinity","-1e2","0","+.5E-2"}) assert(floating(value));
 for(auto value:{"1.","1e+","NaNx","","-.","1.2.3"}) assert(!floating(value));
 assert(genotypes(2,2)==3);
 assert(genotypes(3,2)==6);
 assert(genotypes(3,3)==10);
 assert(genotypes(4,4)==35);
 assert(genotypes(1,300)==1);
 std::cout<<"VCF_HEADER_REFERENCE_OK\n";
}


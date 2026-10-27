// Independent C++23 oracle for the VCF core/GT fixtures. Header/INFO/allele
// semantics are outside this record-level contract.
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
using I=std::int64_t;
static std::vector<std::string_view> split(std::string_view s,char delimiter) {
    std::vector<std::string_view> out;
    for(;;){const auto p=s.find(delimiter);out.push_back(s.substr(0,p));if(p==s.npos)break;s.remove_prefix(p+1);}
    return out;
}
static bool utf8(std::string_view s) {
    for(std::size_t i=0;i<s.size();) {
        const auto a=static_cast<unsigned char>(s[i++]);if(a<128)continue;
        int remaining=0;unsigned char low=128,high=191;
        if(a>=194&&a<=223)remaining=1;
        else if(a>=224&&a<=239){remaining=2;if(a==224)low=160;if(a==237)high=159;}
        else if(a>=240&&a<=244){remaining=3;if(a==240)low=144;if(a==244)high=143;}
        else return false;
        if(s.size()-i<static_cast<std::size_t>(remaining))return false;
        auto b=static_cast<unsigned char>(s[i++]);if(b<low||b>high)return false;
        while(--remaining){b=static_cast<unsigned char>(s[i++]);if(b<128||b>191)return false;}
    }
    return true;
}
static int integer(std::string_view s,I& value) {
    if(s.empty())return 22;
    for(const char c:s)if(c<'0'||c>'9')return 22;
    const auto r=std::from_chars(s.data(),s.data()+s.size(),value);
    return r.ec==std::errc::result_out_of_range?75:r.ec==std::errc{}&&r.ptr==s.data()+s.size()?0:22;
}
struct Result{int status=0;I position=0,columns=0,alternates=0,samples=0;};
static Result parse(std::string_view line) {
    if(!utf8(line))return {84};
    if(line.starts_with("\xef\xbb\xbf"))return {22};
    if(line.ends_with("\n")){line.remove_suffix(1);if(line.ends_with("\r"))line.remove_suffix(1);}
    if(line.empty()||line.front()=='#')return {22};
    for(const auto c:line)if((static_cast<unsigned char>(c)<=32&&c!='\t')||c==127)return {22};
    const auto fields=split(line,'\t');
    if(fields.size()<8||fields.size()==9)return {22};
    for(const auto f:fields)if(f.empty())return {22};
    if(fields[0]==".")return {22};
    Result r;
    if(const int code=integer(fields[1],r.position))return {code};
    for(const char c:fields[3])if(std::string_view("ACGTNacgtn").find(c)==std::string_view::npos)return {22};
    if(fields[4]!=".") {
        const auto alts=split(fields[4],',');
        for(const auto a:alts)if(a.empty()||a==".")return {22};
        r.alternates=static_cast<I>(alts.size());
    }
    if(fields[5]!=".") {
        auto q=fields[5];if(q.front()=='-')return {22};if(q.front()=='+')q.remove_prefix(1);
        double quality=0;
        const auto parsed=std::from_chars(q.data(),q.data()+q.size(),quality);
        if(parsed.ec!=std::errc{}||parsed.ptr!=q.data()+q.size()||!std::isfinite(quality)||quality<0)return {22};
    }
    if(fields.size()>8){
        const auto format=split(fields[8],':');
        for(std::size_t i=0;i<format.size();++i)if(format[i].empty()||(format[i]=="GT"&&i!=0))return {22};
        if(format[0]=="GT")for(std::size_t i=9;i<fields.size();++i){
            auto gt=fields[i].substr(0,fields[i].find(':'));
            std::size_t start=0;
            for(std::size_t end=0;end<=gt.size();++end)if(end==gt.size()||gt[end]=='/'||gt[end]=='|'){
                const auto token=gt.substr(start,end-start);
                if(token!="."){I allele=0;if(const auto code=integer(token,allele))return {code};if(allele>r.alternates)return {22};}
                start=end+1;
            }
        }
    }
    r.columns=static_cast<I>(fields.size());r.samples=r.columns>9?r.columns-9:0;return r;
}
static void check(std::string_view line,int status,I pos=0,I columns=0,I alts=0,I samples=0) {
    const auto r=parse(line);assert(r.status==status);
    if(status==0){assert(r.position==pos&&r.columns==columns&&r.alternates==alts&&r.samples==samples);}
}
int main() {
    check("chr1\t0\t.\tA\t.\t.\tPASS\t.\n",0,0,8,0,0);
    check("chr1\t42\tid\tAC\tA,T\t1.5e+2\tPASS\tDP=4\tGT:DP\t0/1|2:4\t./.:0\r\n",0,42,11,2,2);
    check("chr1\t9223372036854775807\t.\tN\t<DEL>\t.\t.\t.",0,std::numeric_limits<I>::max(),8,1,0);
    check("chr1\t9223372036854775808\t.\tA\tT\t.\t.\t.",75);
    check("chr1\t-1\t.\tA\tT\t.\t.\t.",22);
    check("chr1\t1\t.\tA\tT\t.\t.\t.\tGT\t0/2",22);
    check("chr1\t1\t.\tA\tT\t.\t.\t.\tGT\t0//1",22);
    check("chr1\t1\t.\tA\tT\t.\t.\t.\tDP:GT\t2:0/1",22);
    check("chr1\t1\t.\tA\tT,\t.\t.\t.",22);
    check("chr1\t1\t.\tA\tT\tNaN\t.\t.",22);
    check("chr1\t1\t.\tA\tT\t-1\t.\t.",22);
    check("chr1\t1\t\tA\tT\t.\t.\t.",22);
    check("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",22);
    check("chr1\t1\t.\tA\tT\t.\t.\t.\nchr1",22);
    check(std::string(1,static_cast<char>(192)),84);
    check(std::string("chr1\t1\t.\tA\tT\t.\t.\tNOTE=")+std::string(10000,'A'),0,1,8,1,0);
    std::cout<<"VCF_REFERENCE_OK\n";
}

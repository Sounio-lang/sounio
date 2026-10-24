// C++23 execution harness: immutable corpus, receipts, canonical substring oracles.
#include <atomic>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/wait.h>
namespace fs = std::filesystem;
std::string read(const fs::path& p) { std::ifstream f(p); return {std::istreambuf_iterator<char>(f), {}}; }
std::string q(const std::string& s) { std::string r="'"; for(char c:s) r+=c=='\''?"'\\''":std::string(1,c); return r+"'"; }
int cmd(const std::string& s) { int r=std::system(s.c_str()); return r<0?127:WIFEXITED(r)?WEXITSTATUS(r):128+WTERMSIG(r); }
std::map<std::string,std::string> statuses(const fs::path& p) {
    std::map<std::string,std::string> out; std::istringstream data(read(p)); std::string line; std::getline(data,line);
    while(std::getline(data,line)) { std::istringstream row(line); std::string name,status; std::getline(row,name,'\t'); std::getline(row,status,'\t'); out[name]=status; }
    return out;
}
int main(int argc,char** argv) {
    if(argc==4 && std::string(argv[1])=="--compare") {
        auto before=statuses(fs::path(argv[2])/"results.tsv"), after=statuses(fs::path(argv[3])/"results.tsv");
        if(before.size()!=after.size()) { std::cerr<<"different corpus sizes\n"; return 2; }
        int bp=0,ap=0,improved=0,regressed=0,changed=0;
        for(auto [name,status]:before) {
            if(!after.contains(name)) { std::cerr<<"different corpus members\n"; return 2; }
            const bool b=status=="pass", a=after.at(name)=="pass"; bp+=b; ap+=a;
            const auto old_source=fs::path(argv[2])/name/"source.sio";
            const auto new_source=fs::path(argv[3])/name/"source.sio";
            if(fs::is_regular_file(old_source) && fs::is_regular_file(new_source) && read(old_source)!=read(new_source)) {
                ++changed; std::cout<<"fixture_changed\t"<<name<<'\n';
            }
            if(a&&!b) { ++improved; std::cout<<"fixed\t"<<name<<'\n'; }
            if(b&&!a) { ++regressed; std::cout<<"regressed\t"<<name<<'\t'<<after.at(name)<<'\n'; }
        }
        std::cout<<"before_pass="<<bp<<" after_pass="<<ap<<" newly_passing="<<improved<<" regressed="<<regressed<<" fixture_changes="<<changed<<" total="<<after.size()<<'\n'; return 0;
    }
    if(argc!=6) { std::cerr<<"usage: corpus COMPILER MANIFEST OUTDIR JOBS TIMEOUT\n"; return 2; }
    const auto compiler=fs::absolute(argv[1]).string(); const fs::path out=fs::absolute(argv[3]);
    fs::create_directories(out); std::vector<std::string> names; std::istringstream manifest(read(argv[2]));
    for(std::string s;std::getline(manifest,s);) if(!s.empty()) names.push_back(s);
    std::set<std::string> unique;
    for(const auto& name:names) {
        if(fs::path(name).filename()!=name || !name.ends_with(".sio") || !unique.insert(name).second) {
            std::cerr<<"invalid or duplicate corpus member: "<<name<<'\n'; return 2;
        }
    }
    if(names.empty() || (std::getenv("CORPUS_REQUIRE_200") && names.size()!=200)) {
        std::cerr<<"empty corpus or incorrect required 200-member denominator\n"; return 2;
    }
    const bool replay=std::getenv("CORPUS_REPLAY")!=nullptr;
    if(!replay && fs::exists(out/"results.tsv")) {
        std::cerr<<"result directory already contains receipts; use a fresh directory\n"; return 2;
    }
    // Validate source identity BEFORE opening/truncating any result receipt.
    // A manifest pins members, not contents: fixture edits invalidate replay.
    for(const auto& name:names) {
        const auto src=fs::path("tests/run-pass")/name;
        if(!fs::is_regular_file(src)) { std::cerr<<"missing fixture: "<<name<<'\n'; return 2; }
        if(replay && (!fs::is_regular_file(out/name/"source.sio") || read(src)!=read(out/name/"source.sio"))) {
            std::cerr<<"fixture source changed; cannot replay receipts: "<<name<<'\n'; return 2;
        }
    }
    std::map<std::string,std::pair<int,int>> prior;
    if(replay) { std::istringstream old(read(out/"results.tsv")); std::string line; std::getline(old,line);
        while(std::getline(old,line)) { std::istringstream row(line); std::string name,status,cr,rr;
            std::getline(row,name,'\t'); std::getline(row,status,'\t'); std::getline(row,cr,'\t'); std::getline(row,rr,'\t');
            prior[name]={std::stoi(cr),std::stoi(rr)};
        }
        if(prior.size()!=names.size()) { std::cerr<<"incomplete receipts; cannot replay\n"; return 2; }
    }
    const auto identity_command="shasum -a 256 "+q(compiler)+" "+q(fs::absolute(argv[2]).string());
    if(replay) {
        if(cmd(identity_command+" >"+q((out/"identity.current.txt").string())) || read(out/"identity.txt")!=read(out/"identity.current.txt")) {
            std::cerr<<"compiler/manifest identity changed; cannot replay receipts\n"; return 2;
        }
    } else if(cmd(identity_command+" >"+q((out/"identity.txt").string()))) return 2;
    std::ofstream results(out/"results.tsv"); results<<"test\tstatus\tcompile_rc\trun_rc\toracle\n";
    std::mutex lock; std::atomic_size_t next=0; std::map<std::string,int> totals;
    const int jobs=std::stoi(argv[4]); const auto timeout=std::to_string(std::stoi(argv[5]));
    std::vector<std::thread> workers;
    for(int w=0;w<jobs;++w) workers.emplace_back([&]{
        for(;;) { size_t i=next++; if(i>=names.size()) break;
            const auto& name=names[i]; auto dir=out/name; fs::create_directories(dir);
            const auto src=fs::path("tests/run-pass")/name; auto bin=(dir/"program").string();
            const auto source_text=read(src);
            if(!replay) std::ofstream(dir/"source.sio")<<source_text;
            std::string expected; std::vector<std::string> contains; int expected_rc=0;
            std::istringstream source(source_text); std::string annotation_error;
            for(std::string line; std::getline(source,line);) {
                const std::string p="//@ expect-stdout: ", c="//@ expect-stdout-contains: ", e="//@ expect-exit-code: ";
                if(line.starts_with(p)) { expected+=line.substr(p.size())+"\n"; contains.push_back(line.substr(p.size())); }
                else if(line.starts_with(c)) contains.push_back(line.substr(c.size()));
                else if(line.starts_with(e)) expected_rc=std::stoi(line.substr(e.size()));
                else if(line.starts_with("//@ expect-")) annotation_error=line;
            }
            std::ofstream(dir/"expected.stdout")<<expected;
            auto redir=[&](std::string s){return " >"+q((dir/(s+".log")).string())+" 2>&1";};
            int cr=replay?prior.at(name).first:cmd("timeout "+timeout+" "+q(compiler)+" build "+q(src.string())+" --target aarch64-macos -o "+q(bin)+redir("build"));
            int rr=-1; std::string status;
            if(cr) status=cr==124?"compile_timeout":"compile_failed";
            else if(cmd("file "+q(bin)+" | grep -Eq 'Mach-O 64-bit.*arm64'")) status="wrong_format";
            else if(cmd((replay?std::string():"chmod +x "+q(bin)+" && codesign -s - -f "+q(bin)+" && ")+"codesign --verify "+q(bin)+redir(replay?"sign-replay":"sign"))) status="sign_failed";
            else if(!replay && cmd("shasum -a 256 "+q(bin)+" >"+q((dir/"program.sha256").string()))) status="identity_failed";
            else {
                rr=replay?prior.at(name).second:cmd("timeout 30 "+q(bin)+" >"+q((dir/"stdout.log").string())+" 2>"+q((dir/"stderr.log").string()));
                const auto actual=read(dir/"stdout.log");
                if(rr==124) status="run_timeout";
                else if(rr!=expected_rc) status="exit_mismatch";
                else if(!annotation_error.empty()) status="annotation_unsupported";
                else { status="pass"; for(auto& s:contains) if(actual.find(s)==std::string::npos) status="stdout_mismatch"; }
                constexpr std::array<std::string_view,10> requires_metal = {
                    "gpu_epistemic_f64_sedenion_parity.sio", "gpu_epistemic_f64_ossm_parity.sio",
                    "gpu_readonly_multi_slice.sio", "gpu_param_names_read_mut.sio",
                    "gpu_param_names_readonly_pair.sio", "gpu_vec_add.sio", "gpu_vec_scalar_ops.sio",
                    "gpu_vec_scale.sio", "gpu_vec_ops.sio", "gpu_vec_unary.sio"
                };
                if(status=="pass" && std::getenv("CORPUS_REQUIRE_METAL") &&
                   std::find(requires_metal.begin(),requires_metal.end(),name)!=requires_metal.end() &&
                   read(dir/"stderr.log").find("SOUNIO_METAL_EXECUTED device=")==std::string::npos)
                    status="metal_not_executed";
            }
            std::lock_guard guard(lock); totals[status]++;
            results<<name<<'\t'<<status<<'\t'<<cr<<'\t'<<rr<<'\t'<<(contains.empty()?"exit-only":"contains")<<'\n'; results.flush();
            std::cout<<i+1<<'/'<<names.size()<<' '<<name<<' '<<status<<std::endl;
            if(status!="pass") {
                std::cerr<<"FAIL "<<name<<" status="<<status<<" compile_rc="<<cr<<" run_rc="<<rr<<'\n';
                for(const auto* log:{"build.log","sign.log","stdout.log","stderr.log"}) {
                    const auto text=read(dir/log);
                    if(!text.empty()) std::cerr<<"--- "<<name<<'/'<<log<<" ---\n"<<text<<'\n';
                }
            }
        }
    });
    for(auto& t:workers)t.join(); std::ofstream summary(out/"summary.txt");
    for(auto [s,n]:totals) { std::cout<<s<<'='<<n<<'\n'; summary<<s<<'='<<n<<'\n'; }
    std::map<std::string,int> causes;
    for(const auto& name:names) { std::string cause; std::istringstream log(read(out/name/"build.log"));
        for(std::string line;std::getline(log,line);) {
            if(line.starts_with("ARM64_UNSUPPORTED")) { auto p=line.find("opcode="); cause=p==std::string::npos?"unsupported":line.substr(p); break; }
            if(line.starts_with("error[E") && cause.empty()) { auto p=line.find(']'); cause=line.substr(0,p+1); }
            if(line.starts_with("ARM64_REFUSAL") && cause.empty()) cause="refused_without_opcode";
        }
        if(!cause.empty()) causes[cause]++;
    }
    std::ofstream causes_file(out/"causes.tsv"); for(auto [c,n]:causes) causes_file<<c<<'\t'<<n<<'\n';
    summary<<"total="<<names.size()<<'\n'; return totals["pass"]==static_cast<int>(names.size())?0:1;
}

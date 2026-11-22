#include <filesystem>
#include <cstdint>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Extract the production allocator, scalar lanes and handle checks unchanged.
// The fixture constructs arena states; it does not implement another allocator.
std::string block(const std::string& s, const std::string& marker) {
    auto p=s.find(marker);
    if(p==std::string::npos) throw std::runtime_error("missing "+marker);
    auto end=s.find('{',p); int depth=1; ++end;
    for(;end<s.size() && depth; ++end) {
        if(s[end]=='{') ++depth;
        if(s[end]=='}') --depth;
    }
    if(depth) throw std::runtime_error("unclosed "+marker);
    return s.substr(p,end-p)+"\n";
}
int main(int argc,char** argv) try {
    if(argc!=3 && argc!=4) throw std::runtime_error("usage: probe ROOT OUTPUT [ADDRESS_OUTPUT]");
    std::ifstream in(std::filesystem::path(argv[1])/"self-hosted/ir/ir.sio");
    if(!in) throw std::runtime_error("missing ir.sio");
    std::string s((std::istreambuf_iterator<char>(in)),{});
    std::ofstream out(argv[2]);
    std::istringstream lines(s); std::string line;
    while(std::getline(lines,line)) {
        if(line.starts_with("pub var IR_A_") || line.starts_with("pub var IR_REGION_") ||
           line.starts_with("pub var IR_ARENA_TOP:") || line.starts_with("pub let IR_REGION_") ||
           line.starts_with("pub let IR_ARENA_") ||
           line.starts_with("pub var IR_HOLE_") || line.starts_with("pub let IR_HOLE_")) out<<line<<'\n';
    }
    for(auto marker:{"pub struct IrInstrRegion {","pub fn ir_instr_arena_capacity()",
       "pub fn ir_region_status_v(","pub fn ir_region_status(","pub fn ir_arena_copy_slot(",
       "fn ir_arena_push_hole(","fn ir_arena_take_hole(",
       "fn ir_region_grow_slot(","pub fn ir_region_reserve("}) out<<block(s,marker);
    out<<R"SIO(
fn main() -> i64 with IO, Mut, Panic, Div {
    IR_REGION_COUNT = 2
    IR_REGION_STATE[1] = IR_REGION_LIVE
    IR_REGION_GEN[1] = 7
    IR_REGION_BASE[1] = 64
    IR_REGION_CAP[1] = 8
    IR_REGION_LEN[1] = 2
    IR_ARENA_TOP = 72
    IR_A_IMM_I64[64] = 123
    IR_A_IMM_I64[65] = 456
    IR_A_ARG_BASE[64] = 81
    IR_A_NAME_ID[65] = 92
    if ir_region_grow_slot(1, 9) != IR_ARENA_OK { return 1 }
    if IR_REGION_BASE[1] != 64 || IR_ARENA_TOP != 80 { return 2 }
    if IR_A_IMM_I64[64] != 123 || IR_A_IMM_I64[65] != 456 { return 3 }
    var r = IrInstrRegion { slot: 1, generation: 7 }
    let stale = r
    if ir_region_reserve(&! r, 17) != IR_ARENA_OK { return 4 }
    if IR_REGION_BASE[1] != 64 || IR_ARENA_TOP != 96 { return 5 }
    if ir_region_status(&stale) != IR_ARENA_ERR_STALE { return 6 }
    if ir_region_status(&r) != IR_ARENA_OK { return 7 }
    IR_REGION_STATE[1] = IR_REGION_SEALED
    if ir_region_reserve(&! r, 33) != IR_ARENA_ERR_SEALED { return 8 }
    IR_REGION_STATE[1] = IR_REGION_LIVE
    IR_REGION_COUNT = 3
    IR_REGION_BASE[2] = 96
    IR_REGION_CAP[2] = 8
    IR_REGION_LEN[2] = 1
    IR_REGION_STATE[2] = IR_REGION_LIVE
    IR_A_IMM_I64[96] = 999
    IR_ARENA_TOP = 104
    if ir_region_reserve(&! r, 33) != IR_ARENA_OK { return 9 }
    if IR_REGION_BASE[1] != 104 || IR_ARENA_TOP != 168 { return 10 }
    if IR_A_IMM_I64[104] != 123 || IR_A_IMM_I64[105] != 456 { return 11 }
    if IR_A_ARG_BASE[104] != 81 || IR_A_NAME_ID[105] != 92 { return 12 }
    if IR_A_IMM_I64[96] != 999 { return 13 }
    let cap = ir_instr_arena_capacity()
    IR_REGION_BASE[1] = cap - 8
    IR_REGION_CAP[1] = 8
    IR_ARENA_TOP = cap
    let gen = IR_REGION_GEN[1]
    if ir_region_reserve(&! r, 9) != IR_ARENA_ERR_CAPACITY { return 14 }
    if IR_ARENA_TOP != cap || IR_REGION_CAP[1] != 8 { return 15 }
    if IR_REGION_GEN[1] != gen { return 16 }
    IR_REGION_BASE[1] = cap - 16
    IR_ARENA_TOP = cap - 8
    if ir_region_reserve(&! r, 9) != IR_ARENA_OK { return 17 }
    if IR_ARENA_TOP != cap || IR_REGION_BASE[1] != cap - 16 { return 18 }
    if ir_region_reserve(&! r, -1) != IR_ARENA_ERR_NEGATIVE { return 19 }
    if ir_region_grow_slot(-1, 9) != IR_ARENA_ERR_INVALID { return 20 }
    println("IR_TAIL_GROWTH_OK")
    0
}
)SIO";
    if(!out) throw std::runtime_error("write failed");
    if(argc==4) {
        std::ifstream backend(std::filesystem::path(argv[1])/"self-hosted/native/codegen_arm64_macos.sio");
        if(!backend) throw std::runtime_error("missing ARM64 backend");
        std::string b((std::istreambuf_iterator<char>(backend)),{});
        std::ofstream addr(argv[3]);
        std::istringstream decls(b);
        while(std::getline(decls,line))
            if(line.starts_with("var A64M_CODE:") || line.starts_with("var A64M_LEN:") ||
               line.starts_with("var A64M_BSS:") || line.starts_with("var A64M_ERROR:")) addr<<line<<'\n';
        for(auto marker:{"fn a64m_byte(","fn a64m_word(","fn a64m_data_addr("}) addr<<block(b,marker);
        addr<<"fn main() -> i64 with IO, Mut, Panic, Div {\n";
        constexpr std::int64_t limit=6442450944LL-4429185024LL-32768;
        int code=30;
        for(std::int64_t offset:std::array<std::int64_t,6>{-32,0,134217720,134217728,571090960,limit-8}) {
            auto target=4429185024LL+16384+offset;
            auto delta=target/4096-(4294967296LL+16384)/4096;
            auto adrp=0x90000000LL|((delta&3)<<29)|(((delta>>2)&524287)<<5)|9;
            auto add=0x91000000LL|((target&4095)<<10)|(9<<5)|9;
            addr<<" A64M_LEN = 0\n A64M_ERROR = false\n a64m_data_addr(9, "<<offset<<")\n"
                <<" if A64M_ERROR || A64M_LEN != 8 { return "<<code++<<" }\n";
            for(int i=0;i<8;++i) {
                auto word=i<4?adrp:add;
                auto byte=(word>>((i%4)*8))&255;
                addr<<" if (A64M_CODE["<<i<<"] as i64 & 255) != "<<byte<<" { return "<<code++<<" }\n";
            }
        }
        for(std::int64_t offset:std::array<std::int64_t,2>{-33,limit-7}) {
            addr<<" A64M_LEN = 0\n A64M_ERROR = false\n a64m_data_addr(9, "<<offset<<")\n"
                <<" if !A64M_ERROR || A64M_LEN != 0 { return "<<code++<<" }\n";
        }
        addr<<" println(\"ARM64_DATA_ADDRESS_OK\")\n 0\n}\n";
        if(!addr) throw std::runtime_error("address write failed");
    }
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }

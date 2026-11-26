// Independent offline reference for the Granite-3.1-8B-instruct audit.
// Input tables are extracted from the pinned tokenizer.json with jq (see
// granite_reference.sbatch). PCRE2 implements the ByteLevel GPT-2 regex;
// this tool is never in the Sounio inference path.
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace {
using ID = uint32_t;
constexpr ID absent = std::numeric_limits<ID>::max();
std::string decode64(std::string_view s) {
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    uint32_t bits = 0;
    unsigned n = 0;
    std::string out;
    for (char c : s) {
        if (c == '=') break;
        auto value = alphabet.find(c);
        if (value == alphabet.npos) throw std::runtime_error("invalid base64");
        bits = (bits << 6) | static_cast<uint32_t>(value);
        n += 6;
        if (n >= 8) { n -= 8; out.push_back(static_cast<char>((bits >> n) & 255)); }
    }
    return out;
}
std::string utf8(unsigned cp) {
    if (cp < 128) return std::string(1, static_cast<char>(cp));
    if (cp >= 2048) throw std::runtime_error("unexpected byte alphabet codepoint");
    return {static_cast<char>(0xc0 | (cp >> 6)), static_cast<char>(0x80 | (cp & 63))};
}
std::array<std::string, 256> byte_alphabet() {
    std::array<std::string, 256> result;
    unsigned next = 256;
    for (unsigned b = 0; b < 256; ++b) {
        bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        result[b] = utf8(direct ? b : next++);
    }
    return result;
}
uint64_t key(ID a, ID b) { return (static_cast<uint64_t>(a) << 32) | b; }
struct Merge { size_t rank; ID product; };
struct Tokenizer {
    std::unordered_map<std::string, ID> vocab;
    std::unordered_map<uint64_t, Merge> merges;
    std::vector<std::pair<std::string, ID>> specials;
    std::array<ID, 256> bytes{};
    pcre2_code* regex = nullptr;
    Tokenizer() {
        constexpr std::string_view pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
        int err = 0; PCRE2_SIZE offset = 0;
        regex = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF | PCRE2_UCP, &err, &offset, nullptr);
        if (!regex) throw std::runtime_error("PCRE2 regex compile failure");
    }
    ~Tokenizer() { pcre2_code_free(regex); }
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    ID lookup(const std::string& s) const {
        auto it = vocab.find(s);
        if (it == vocab.end()) throw std::runtime_error("missing vocabulary entry");
        return it->second;
    }
    void load(const char* vp, const char* mp, const char* sp) {
        std::ifstream v(vp), m(mp), s(sp);
        if (!v || !m || !s) throw std::runtime_error("cannot open tokenizer tables");
        std::string a, b;
        ID id;
        while (v >> a >> id) {
            if (id >= 49155 || !vocab.emplace(decode64(a), id).second) throw std::runtime_error("invalid vocab");
        }
        if (!v.eof()) throw std::runtime_error("invalid vocab record");
        size_t rank = 0;
        while (m >> a >> b) {
            a = decode64(a); b = decode64(b);
            if (!merges.emplace(key(lookup(a), lookup(b)), Merge{rank++, lookup(a + b)}).second) throw std::runtime_error("duplicate merge");
        }
        if (!m.eof()) throw std::runtime_error("invalid merge record");
        while (s >> a >> id) {
            if (id >= 49155) throw std::runtime_error("special outside vocabulary");
            specials.emplace_back(decode64(a), id);
        }
        if (!s.eof() || vocab.empty() || merges.empty() || specials.empty()) throw std::runtime_error("incomplete tokenizer tables");
        auto alphabet = byte_alphabet();
        for (size_t i = 0; i < alphabet.size(); ++i) {
            auto it = vocab.find(alphabet[i]);
            bytes[i] = it == vocab.end() ? absent : it->second;
        }
    }
    void bpe(std::string_view text, std::vector<ID>& out) const {
        std::vector<ID> seq;
        for (unsigned char c : text) {
            if (bytes[c] == absent) throw std::runtime_error("input byte absent from Granite vocabulary: " + std::to_string(c));
            seq.push_back(bytes[c]);
        }
        while (seq.size() > 1) {
            size_t best = std::numeric_limits<size_t>::max(), at = 0;
            ID product = absent;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                auto it = merges.find(key(seq[i], seq[i+1]));
                if (it != merges.end() && it->second.rank < best) { best = it->second.rank; at = i; product = it->second.product; }
            }
            if (product == absent) break;
            seq[at] = product;
            seq.erase(seq.begin() + static_cast<std::ptrdiff_t>(at + 1));
        }
        out.insert(out.end(), seq.begin(), seq.end());
    }
    void segment(std::string_view text, std::vector<ID>& out) const {
        if (text.empty()) return;
        auto* match = pcre2_match_data_create_from_pattern(regex, nullptr);
        if (!match) throw std::runtime_error("PCRE2 allocation failure");
        for (size_t pos = 0; pos < text.size();) {
            auto rc = pcre2_match(regex, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), pos, PCRE2_ANCHORED, match, nullptr);
            if (rc < 0) { pcre2_match_data_free(match); throw std::runtime_error("ByteLevel regex failed at byte " + std::to_string(pos)); }
            auto* offsets = pcre2_get_ovector_pointer(match);
            if (offsets[0] != pos || offsets[1] <= pos) { pcre2_match_data_free(match); throw std::runtime_error("ByteLevel regex did not advance"); }
            bpe(text.substr(pos, offsets[1] - pos), out);
            pos = offsets[1];
        }
        pcre2_match_data_free(match);
    }
    std::vector<ID> encode(std::string_view text) const {
        std::vector<ID> out;
        size_t start = 0, pos = 0;
        while (pos < text.size()) {
            const std::pair<std::string, ID>* best = nullptr;
            for (const auto& special : specials)
                if (text.substr(pos).starts_with(special.first) && (!best || special.first.size() > best->first.size())) best = &special;
            if (!best) { ++pos; continue; }
            segment(text.substr(start, pos - start), out);
            out.push_back(best->second);
            pos += best->first.size(); start = pos;
        }
        segment(text.substr(start), out);
        return out;
    }
};
std::string chat(const std::string& user) {
    return "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\n"
           "Today's Date: November 26, 2026.\nYou are Granite, developed by IBM. You are a helpful AI assistant."
           "<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>" + user +
           "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: granite_prompt_ids vocab.tsv merges.tsv specials.tsv prompts.txt");
        Tokenizer tok;
        tok.load(argv[1], argv[2], argv[3]);
        if (std::string_view(argv[4]) == "--smoke") {
            // The pinned Granite vocabulary does not contain all 256 ByteLevel symbols.
            // Accept sparse tables, but fail closed on an unrepresentable input byte.
            if (std::count(tok.bytes.begin(), tok.bytes.end(), absent) != 14)
                throw std::runtime_error("unexpected Granite byte alphabet coverage");
            bool missing_rejected = false;
            try { std::vector<ID> ids; tok.bpe(std::string(1, static_cast<char>(192)), ids); }
            catch (const std::runtime_error& e) { missing_rejected = std::string_view(e.what()).starts_with("input byte absent"); }
            if (!missing_rejected) throw std::runtime_error("missing input byte not rejected");
            if (tok.encode("hello") != std::vector<ID>{7656} ||
                tok.encode(" hello!") != std::vector<ID>{17964, 19} ||
                tok.encode("123") != std::vector<ID>{35, 36, 37} ||
                tok.encode("<|start_of_role|><|end_of_role|><|end_of_text|><|tool_call|>") != std::vector<ID>{49152, 49153, 0, 49154})
                throw std::runtime_error("Granite fixed vocabulary controls failed");
            std::cout << "GRANITE_CPP_TOKENIZER_SMOKE_PASS\n";
            return 0;
        }
        std::ifstream prompts(argv[4]);
        if (!prompts) throw std::runtime_error("cannot open prompts");
        std::ostringstream result;
        std::string line;
        size_t count = 0;
        while (std::getline(prompts, line)) {
            while (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            auto ids = tok.encode(chat(line));
            if (ids.empty()) throw std::runtime_error("empty prompt encoding");
            for (size_t i = 0; i < ids.size(); ++i) result << (i ? " " : "") << ids[i];
            result << '\n';
            ++count;
        }
        if (count != 480) throw std::runtime_error("expected exactly 480 nonempty prompts, got " + std::to_string(count));
        std::cout << result.str();
        return std::cout ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

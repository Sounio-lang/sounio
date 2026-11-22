// Bounded analysis of strace -f -q -yy -xx, PID-prefixed pathname/process
// records. Produces private byte-preserving event ledgers; never a cache
// dependency manifest.
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
string trim(string s) {
  auto a = s.find_first_not_of(" \t\r\n");
  if (a == s.npos)
    return {};
  return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
string hex(const string &s) {
  const char *d = "0123456789abcdef";
  string r;
  for (unsigned char c : s) {
    r += d[c >> 4];
    r += d[c & 15];
  }
  return r;
}
int nib(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}
optional<string> decode_hex_token(const string &s) {
  if (s.size() < 2 || s.front() != '"' || s.back() != '"')
    return nullopt;
  string r;
  for (size_t i = 1; i + 1 < s.size();) {
    if (i + 3 >= s.size() - 1 || s[i] != '\\' || s[i + 1] != 'x' ||
        nib(s[i + 2]) < 0 || nib(s[i + 3]) < 0)
      return nullopt;
    r += char(nib(s[i + 2]) * 16 + nib(s[i + 3]));
    i += 4;
  }
  return r;
}
vector<string> args(const string &s) {
  vector<string> r;
  int depth = 0;
  bool q = false, esc = false;
  size_t start = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (q) {
      if (esc)
        esc = false;
      else if (c == '\\')
        esc = true;
      else if (c == '"')
        q = false;
      continue;
    }
    if (c == '"')
      q = true;
    else if (c == '(' || c == '[' || c == '{')
      ++depth;
    else if (c == ')' || c == ']' || c == '}')
      --depth;
    else if (c == ',' && depth == 0) {
      r.push_back(trim(s.substr(start, i - start)));
      start = i + 1;
    }
  }
  r.push_back(trim(s.substr(start)));
  return r;
}
size_t closing_paren(const string &s, size_t start) {
  if (start == string::npos)
    return string::npos;
  int depth = 0;
  bool q = false, esc = false;
  for (size_t i = start; i < s.size(); ++i) {
    char c = s[i];
    if (q) {
      if (esc)
        esc = false;
      else if (c == 92)
        esc = true;
      else if (c == 34)
        q = false;
      continue;
    }
    if (c == 34)
      q = true;
    else if (c == '(')
      ++depth;
    else if (c == ')' && --depth == 0)
      return i;
  }
  return string::npos;
}
struct Pending {

  string text;
  size_t line;
};
struct Image {
  string path;
  size_t epoch = 0;
};
struct Count {
  size_t events = 0, failures = 0, operands = 0;
};
int main(int argc, char **argv) {
  try {
    if (argc < 4) {
      cerr << "usage: analyzer TRACE NEW_OUTPUT_DIRECTORY SOURCE_ROOT "
              "[EXACT_COMPILER_IMAGE...]\n";
      return 64;
    }
    ifstream in(argv[1]);
    if (!in)
      throw runtime_error("trace unreadable");
    filesystem::path out(argv[2]);
    if (!filesystem::create_directory(out))
      throw runtime_error("output must be a new directory");
    filesystem::permissions(out, filesystem::perms::owner_all,
                            filesystem::perm_options::replace);
    ofstream events(out / "events.tsv"), execs(out / "exec.tsv"),
        paths(out / "path_operands.tsv"), summary(out / "SUMMARY.txt"),
        rejected(out / "unresolved.tsv");
    if (!events || !execs || !paths || !summary || !rejected)
      throw runtime_error("output unavailable");
    set<string> comp;
    for (int i = 4; i < argc; ++i)
      comp.insert(argv[i]);
    map<string, size_t> epochs;
    map<string, Pending> pending;
    map<string, Image> images;
    map<pair<string, string>, Count> counts;
    map<string, set<string>> candidates;
    size_t lines = 0, complete = 0, joined = 0, unparsed = 0, orphan = 0,
           replaced = 0, decodefail = 0, unknown = 0, failedexec = 0;
    string root = argv[3];
    while (root.size() > 1 && root.back() == '/')
      root.pop_back();
    events << "first_line\tlast_line\tpid\tepoch\timage_hex\trole_"
              "hint\tfailed\tsyscall_hex\trecord_hex\n";
    execs << "first_line\tlast_line\tpid\tepoch_after\tsuccess\trequested_"
             "image_hex\n";
    paths << "first_line\tlast_line\tpid\tepoch\trole_hint\tsyscall_"
             "hex\toperand_index\tpath_hex\targument_context_hex\tfailed\n";
    auto role = [&](const string &image) {
      if (comp.contains(image))
        return string("compiler_image");
      if (image == "/usr/bin/sha256sum")
        return string("hash_utility_image");
      return image.empty() ? string("unknown") : string("other_image");
    };
    // Only these explicitly named pathname positions are decoded. Other
    // events stay in events.tsv.
    const map<string, vector<int>> positions = {
        {"open", {0}},        {"openat", {1}},       {"openat2", {1}},
        {"access", {0}},      {"faccessat", {1}},    {"faccessat2", {1}},
        {"stat", {0}},        {"lstat", {0}},        {"newfstatat", {1}},
        {"statx", {1}},       {"readlink", {0}},     {"readlinkat", {1}},
        {"execve", {0}},      {"execveat", {1}},     {"chdir", {0}},
        {"unlink", {0}},      {"unlinkat", {1}},     {"rename", {0, 1}},
        {"renameat", {1, 3}}, {"renameat2", {1, 3}}, {"mkdir", {0}},
        {"mkdirat", {1}},     {"rmdir", {0}},        {"chmod", {0}},
        {"fchmodat", {1}},    {"chown", {0}},        {"lchown", {0}},
        {"fchownat", {1}},    {"link", {0, 1}},      {"linkat", {1, 3}},
        {"symlink", {1}},     {"symlinkat", {2}},    {"utimensat", {1}}};
    rejected << "line\treason\trecord_hex\n";
    string line;
    while (true) {
      line.clear();
      char byte;
      while (in.get(byte) && byte != '\n') {
        if (line.size() >= 1048576)
          throw runtime_error("line bound exceeded; output partial");
        line += byte;
      }
      if (line.empty() && !in)
        break;
      ++lines;
      if (lines > 2000000 || line.size() > 1048576)
        throw runtime_error("trace bound exceeded; output partial");
      auto sp = line.find(' ');
      if (sp == string::npos) {
        ++unparsed;
        rejected << lines << "\tunparsed\t" << hex(line) << '\n';
        continue;
      }
      string pid = line.substr(0, sp);
      unsigned long long p = 0;
      auto parse = from_chars(pid.data(), pid.data() + pid.size(), p);
      if (parse.ec != errc{} || parse.ptr != pid.data() + pid.size()) {
        ++unparsed;
        rejected << lines << "\tunparsed\t" << hex(line) << '\n';
        continue;
      }
      if (images.size() > 100000 || pending.size() > 100000)
        throw runtime_error("task bound exceeded; output partial");
      string s = trim(line.substr(sp + 1));
      size_t first = lines;
      if (s.rfind("<... ", 0) == 0) {
        auto e = s.find(" resumed>");
        auto it = pending.find(pid);
        if (e == string::npos || it == pending.end()) {
          ++orphan;
          rejected << lines << "\torphan_or_mismatch\t" << hex(line) << '\n';
          continue;
        }
        auto name = s.substr(5, e - 5);
        if (it->second.text.rfind(name + "(", 0) != 0) {
          ++orphan;
          rejected << lines << "\torphan_or_mismatch\t" << hex(line) << '\n';
          continue;
        }
        first = it->second.line;
        s = it->second.text + s.substr(e + 9);
        pending.erase(it);
        ++joined;
      }
      auto u = s.find("<unfinished ...>");
      if (u != s.npos) {
        if (pending.contains(pid)) {
          ++replaced;
          rejected << pending[pid].line << "\treplaced_pending\t"
                   << hex(pending[pid].text) << '\n';
        }
        pending[pid] = {s.substr(0, u), first};
        continue;
      }
      if (s.rfind("+++ ", 0) == 0) {
        images.erase(pid);
        ++complete;
        continue;
      }
      if (s.rfind("--- ", 0) == 0) {
        ++complete;
        continue;
      }
      auto a = s.find('('), b = closing_paren(s, a);
      if (a == s.npos || b == s.npos || b < a) {
        ++unparsed;
        rejected << lines << "\tunparsed\t" << hex(line) << '\n';
        continue;
      }
      auto eq = s.find('=', b + 1);
      if (eq == s.npos) {
        ++unparsed;
        rejected << lines << "\tmissing_result\t" << hex(line) << '\n';
        continue;
      }
      string name = s.substr(0, a), result = trim(s.substr(eq + 1));
      auto av = args(s.substr(a + 1, b - a - 1));
      bool failed = result.rfind("-1 ", 0) == 0;
      bool exec = name == "execve" || name == "execveat";
      bool success = result == "0";
      if (exec) {
        size_t i = name == "execve" ? 0 : 1;
        auto path = i < av.size() ? decode_hex_token(av[i]) : nullopt;
        if (!path) {
          ++decodefail;
          rejected << first << "\tpath_decode_failure\t" << hex(s) << '\n';
        }
        if (success) {
          auto &im = images[pid];
          im.epoch = ++epochs[pid];
          im.path = path ? *path : "";
        } else
          ++failedexec;
        execs << first << '\t' << lines << '\t' << pid << '\t'
              << images[pid].epoch << '\t' << success << '\t'
              << (path ? hex(*path) : "") << '\n';
      }
      auto im = images[pid];
      string r = role(im.path);
      if (im.path.empty())
        ++unknown;
      auto &c = counts[{r, im.path}];
      ++c.events;
      if (failed)
        ++c.failures;
      events << first << '\t' << lines << '\t' << pid << '\t' << im.epoch
             << '\t' << hex(im.path) << '\t' << r << '\t' << failed << '\t'
             << hex(name) << '\t' << hex(s) << '\n';
      auto pos = positions.find(name);
      if (pos != positions.end())
        for (int i : pos->second) {
          auto path = size_t(i) < av.size() ? decode_hex_token(av[i]) : nullopt;
          if (!path) {
            ++decodefail;
            rejected << first << "\tpath_decode_failure\t" << hex(s) << '\n';
            continue;
          }
          ++c.operands;
          paths << first << '\t' << lines << '\t' << pid << '\t' << im.epoch
                << '\t' << r << '\t' << hex(name) << '\t' << i << '\t'
                << hex(*path) << '\t' << hex(s.substr(a + 1, b - a - 1)) << '\t'
                << failed << '\n';
          if (path->rfind(root + "/", 0) == 0)
            candidates[r].insert(path->substr(root.size() + 1));
        }
      ++complete;
    }
    for (auto &[pid, v] : pending)
      rejected << v.line << "\tpending_eof\t" << hex(v.text) << '\n';
    if (in.bad())
      throw runtime_error("trace read failure");
    summary << "schema=1\ncandidate_only=true\ncache_authority=false\ncomplete_"
               "input_closure=false\nrole_policy=exact_exec_path_hint_no_"
               "inheritance\nrelative_path_resolution=none\npath_subset="
               "explicit_whitelist\nlines="
            << lines << "\ncomplete_records=" << complete
            << "\njoined_records=" << joined
            << "\nunparsed_records=" << unparsed
            << "\norphan_resumes=" << orphan
            << "\nreplaced_pending=" << replaced
            << "\npending_at_eof=" << pending.size()
            << "\npath_decode_failures=" << decodefail
            << "\nunknown_image_events=" << unknown
            << "\nunsuccessful_exec_records=" << failedexec << "\n";
    summary << "role_hint\timage_hex\tsyscall_records\tfailed_minus1\tdecoded_"
               "path_operands\n";
    for (auto &[k, c] : counts)
      summary << k.first << '\t' << hex(k.second) << '\t' << c.events << '\t'
              << c.failures << '\t' << c.operands << '\n';
    map<string, Count> totals;
    for (auto &[k, c] : counts) {
      totals[k.first].events += c.events;
      totals[k.first].failures += c.failures;
      totals[k.first].operands += c.operands;
    }
    summary << "ROLE_TOTALS\n";
    for (auto &[r, c] : totals)
      summary << r << '\t' << c.events << '\t' << c.failures << '\t'
              << c.operands << '\n';
    summary << "ABSOLUTE_SOURCE_LEXICAL_CANDIDATES\n";
    for (auto &[r, ps] : candidates)
      summary << r << '\t' << ps.size() << '\n';
    ofstream cand(out / "absolute_source_candidates.tsv");
    cand << "role_hint\trelative_path_hex\n";
    for (auto &[r, ps] : candidates)
      for (auto &p : ps)
        cand << r << '\t' << hex(p) << '\n';
    events.flush();
    execs.flush();
    paths.flush();
    summary.flush();
    cand.flush();
    rejected.flush();
    if (!events || !execs || !paths || !summary || !cand || !rejected)
      throw runtime_error("output write failure");
    return 0;
  } catch (const exception &e) {
    cerr << e.what() << '\n';
    return 70;
  }
}

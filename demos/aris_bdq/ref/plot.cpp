// plot.cpp — minimal SVG line plots (C++23), no external dependencies.
// usage: plot fig7 nom.csv out.svg | plot fig8 nom.csv out.svg | plot band b0.csv b1.csv nom.csv out.svg
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using Col = std::vector<double>;
struct Csv { std::vector<std::string> h; std::vector<Col> c; };

Csv load(const char* f) {
  Csv d; std::ifstream in(f); std::string line;
  std::getline(in, line);
  { std::stringstream s(line); std::string t; while (std::getline(s, t, ',')) d.h.push_back(t); }
  d.c.resize(d.h.size());
  while (std::getline(in, line)) {
    std::stringstream s(line); std::string t; size_t i = 0;
    while (std::getline(s, t, ',') && i < d.c.size()) d.c[i++].push_back(std::stod(t));
  }
  return d;
}
const Col& col(const Csv& d, const std::string& n) {
  for (size_t i = 0; i < d.h.size(); ++i) if (d.h[i] == n) return d.c[i];
  std::fprintf(stderr, "no column %s\n", n.c_str()); std::exit(1);
}

struct Ax {
  double x0 = 80, y0 = 40, w = 620, hgt = 400, xmin, xmax, ymin, ymax; bool logy;
  double X(double x) const { return x0 + (x - xmin) / (xmax - xmin) * w; }
  double Y(double y) const {
    if (logy) { y = std::max(y, ymin); return y0 + hgt - (std::log10(y) - std::log10(ymin)) / (std::log10(ymax) - std::log10(ymin)) * hgt; }
    return y0 + hgt - (y - ymin) / (ymax - ymin) * hgt;
  }
};

std::string path(const Ax& a, const Col& x, const Col& y) {
  std::string s; char b[64];
  for (size_t i = 0; i < x.size(); ++i) { std::snprintf(b, sizeof b, "%c%.1f %.1f", i ? 'L' : 'M', a.X(x[i]), a.Y(y[i])); s += b; }
  return s;
}

void frame(FILE* f, const Ax& a, const char* title, const char* ylab) {
  std::fprintf(f, "<rect x='%g' y='%g' width='%g' height='%g' fill='none' stroke='#888'/>\n", a.x0, a.y0, a.w, a.hgt);
  std::fprintf(f, "<text x='%g' y='24' font-size='15' font-weight='600'>%s</text>\n", a.x0, title);
  std::fprintf(f, "<text x='18' y='%g' font-size='12' transform='rotate(-90 18 %g)' text-anchor='middle'>%s</text>\n", a.y0 + a.hgt / 2, a.y0 + a.hgt / 2, ylab);
  std::fprintf(f, "<text x='%g' y='%g' font-size='12' text-anchor='middle'>Time (h) — weeks 0–24</text>\n", a.x0 + a.w / 2, a.y0 + a.hgt + 36);
  for (double x = 0; x <= a.xmax + 1; x += 500) {
    std::fprintf(f, "<line x1='%.1f' x2='%.1f' y1='%g' y2='%g' stroke='#eee'/>", a.X(x), a.X(x), a.y0, a.y0 + a.hgt);
    std::fprintf(f, "<text x='%.1f' y='%g' font-size='11' text-anchor='middle'>%g</text>\n", a.X(x), a.y0 + a.hgt + 16, x);
  }
  if (a.logy) {
    for (double e = std::floor(std::log10(a.ymin)); e <= std::log10(a.ymax) + 1e-9; e += 1) {
      double y = std::pow(10, e); if (y < a.ymin) continue;
      std::fprintf(f, "<line x1='%g' x2='%g' y1='%.1f' y2='%.1f' stroke='#eee'/>", a.x0, a.x0 + a.w, a.Y(y), a.Y(y));
      std::fprintf(f, "<text x='%g' y='%.1f' font-size='11' text-anchor='end'>%g</text>\n", a.x0 - 6, a.Y(y) + 4, y);
    }
  } else {
    double step = (a.ymax - a.ymin) / 7;
    double mag = std::pow(10, std::floor(std::log10(step))); step = std::ceil(step / mag) * mag;
    for (double y = a.ymin; y <= a.ymax + 1e-9; y += step) {
      std::fprintf(f, "<line x1='%g' x2='%g' y1='%.1f' y2='%.1f' stroke='#eee'/>", a.x0, a.x0 + a.w, a.Y(y), a.Y(y));
      std::fprintf(f, "<text x='%g' y='%.1f' font-size='11' text-anchor='end'>%g</text>\n", a.x0 - 6, a.Y(y) + 4, y);
    }
  }
}
void hline(FILE* f, const Ax& a, double y, const char* lab) {
  std::fprintf(f, "<line x1='%g' x2='%g' y1='%.1f' y2='%.1f' stroke='#c0392b' stroke-width='1.3'/>", a.x0, a.x0 + a.w, a.Y(y), a.Y(y));
  std::fprintf(f, "<text x='%g' y='%.1f' font-size='11' fill='#c0392b'>%s</text>\n", a.x0 + a.w + 4, a.Y(y) + 4, lab);
}
void line(FILE* f, const Ax& a, const Col& x, const Col& y, const char* color, const char* lab, double ly) {
  std::fprintf(f, "<path d='%s' fill='none' stroke='%s' stroke-width='1'/>\n", path(a, x, y).c_str(), color);
  std::fprintf(f, "<text x='%g' y='%.1f' font-size='11' fill='%s'>%s</text>\n", a.x0 + a.w + 4, ly, color, lab);
}
FILE* open(const char* o, int W = 820, int H = 500) {
  FILE* f = std::fopen(o, "w");
  std::fprintf(f, "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' font-family='Helvetica,Arial,sans-serif'>\n<rect width='100%%' height='100%%' fill='white'/>\n", W, H);
  return f;
}

int main(int argc, char** argv) {
  std::string m = argv[1];
  if (m == "fig7") {
    Csv d = load(argv[2]); FILE* f = open(argv[3]);
    Ax a; a.xmin = 0; a.xmax = 4032; a.ymin = 0.01; a.ymax = 1000; a.logy = true;
    frame(f, a, "Fig. 7A reproduced — cellular lesion, human", "Unbound conc. (ng/mL or ng/g)");
    const Col& t = col(d, "t");
    line(f, a, t, col(d, "lys_m"), "#1f4e9c", "Lysosome", a.Y(200));
    line(f, a, t, col(d, "cyt_m"), "#4f8fd6", "Cytosol", a.Y(20));
    line(f, a, t, col(d, "cp_u"), "#7aa7d9", "Plasma", a.Y(0.6));
    hline(f, a, 120, "MIC90"); hline(f, a, 60, "MIC50");
    std::fprintf(f, "</svg>\n"); std::fclose(f);
  } else if (m == "fig8") {
    Csv d = load(argv[2]); FILE* f = open(argv[3]);
    Ax a; a.xmin = 0; a.xmax = 4032; a.ymin = 0; a.ymax = 7000; a.logy = false;
    frame(f, a, "Fig. 8 reproduced — total caseum, human", "Total conc. (ng/g)");
    const Col& t = col(d, "t");
    const char* cs[6] = {"#0b3d91", "#1f5fae", "#3b7fc4", "#5e9bd3", "#83b5e0", "#a9cdee"};
    for (int k = 0; k < 6; ++k) {
      std::string n = "cas" + std::to_string(k + 1);
      const Col& y = col(d, n);
      std::string lab = "Caseum-" + std::to_string(k + 1);
      line(f, a, t, y, cs[k], lab.c_str(), a.Y(y.back()) + 4);
    }
    std::fprintf(f, "<line x1='%g' x2='%g' y1='%.1f' y2='%.1f' stroke='#c0392b' stroke-width='1.3'/><text x='%g' y='%.1f' font-size='11' fill='#c0392b'>WCC90</text>\n", a.x0, a.x0 + a.w, a.Y(2780), a.Y(2780), a.x0 + 8, a.Y(2780) + 14);
    std::fprintf(f, "</svg>\n"); std::fclose(f);
  } else if (m == "band") {
    Csv b0 = load(argv[2]), b1 = load(argv[3]), d = load(argv[4]); FILE* f = open(argv[5], 900, 500);
    Ax a; a.xmin = 0; a.xmax = 4032; a.ymin = 5; a.ymax = 2000; a.logy = true;
    frame(f, a, "Macrophage lysosome, unbound: uncertainty propagated to humans", "Unbound conc. (ng/g)");
    auto band = [&](const Csv& b, const char* fill, const char* ink, const char* lab, double ly) {
      const Col& t = col(b, "t"); const Col& lo = col(b, "p025"); const Col& hi = col(b, "p975");
      std::string s = path(a, t, hi); char buf[64];
      for (size_t i = t.size(); i-- > 0;) { std::snprintf(buf, sizeof buf, "L%.1f %.1f", a.X(t[i]), a.Y(lo[i])); s += buf; }
      s += "Z";
      std::fprintf(f, "<path d='%s' fill='%s' stroke='none'/>\n", s.c_str(), fill);
      std::fprintf(f, "<text x='%g' y='%.1f' font-size='11' font-weight='600' fill='%s'>%s</text>\n", a.x0 + a.w + 4, ly, ink, lab);
    };
    band(b1, "rgba(231,156,52,0.35)", "#b8701a", "95% CI, with assumptions", a.Y(600));
    band(b0, "rgba(31,78,156,0.45)", "#1f4e9c", "95% CI, reported SEs only", a.Y(300));
    line(f, a, col(d, "t"), col(d, "lys_m"), "#0b2a5c", "Nominal", a.Y(90));
    hline(f, a, 120, "MIC90"); hline(f, a, 60, "MIC50");
    std::fprintf(f, "</svg>\n"); std::fclose(f);
  } else if (m == "pbox") {
    // argv: nominal lo_both hi_both lo_fu hi_fu out.svg — each a file of sorted samples
    auto rd = [](const char* fn) { std::vector<double> v; std::ifstream in(fn); double x; while (in >> x) v.push_back(x); return v; };
    auto v0 = rd(argv[2]), vl = rd(argv[3]), vh = rd(argv[4]), fl = rd(argv[5]), fh = rd(argv[6]);
    FILE* f = open(argv[7], 900, 500);
    Ax a; a.xmin = 0; a.xmax = 1; a.ymin = 0; a.ymax = 1; a.logy = false;
    double lx0 = std::log10(2.0), lx1 = std::log10(2000.0);
    auto X = [&](double c) { return a.x0 + (std::log10(std::max(c, 2.0)) - lx0) / (lx1 - lx0) * a.w; };
    std::fprintf(f, "<rect x='%g' y='%g' width='%g' height='%g' fill='none' stroke='#888'/>\n", a.x0, a.y0, a.w, a.hgt);
    std::fprintf(f, "<text x='%g' y='24' font-size='15' font-weight='600'>P-box: minimum unbound lysosomal concentration, weeks 3–24</text>\n", a.x0);
    for (double c : {2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0}) {
      std::fprintf(f, "<line x1='%.1f' x2='%.1f' y1='%g' y2='%g' stroke='#eee'/><text x='%.1f' y='%g' font-size='11' text-anchor='middle'>%g</text>\n", X(c), X(c), a.y0, a.y0 + a.hgt, X(c), a.y0 + a.hgt + 16, c);
    }
    for (double p = 0; p <= 1.0001; p += 0.25)
      std::fprintf(f, "<line x1='%g' x2='%g' y1='%.1f' y2='%.1f' stroke='#eee'/><text x='%g' y='%.1f' font-size='11' text-anchor='end'>%.2f</text>\n", a.x0, a.x0 + a.w, a.Y(p), a.Y(p), a.x0 - 6, a.Y(p) + 4, p);
    std::fprintf(f, "<text x='%g' y='%g' font-size='12' text-anchor='middle'>Unbound concentration (ng/g), log scale — across virtual patients</text>\n", a.x0 + a.w / 2, a.y0 + a.hgt + 36);
    std::fprintf(f, "<text x='18' y='%g' font-size='12' transform='rotate(-90 18 %g)' text-anchor='middle'>Cumulative probability</text>\n", a.y0 + a.hgt / 2, a.y0 + a.hgt / 2);
    auto cdf = [&](const std::vector<double>& v) {
      std::string s; char b[64];
      for (size_t i = 0; i < v.size(); ++i) { std::snprintf(b, sizeof b, "%c%.1f %.1f", i ? 'L' : 'M', X(v[i]), a.Y((i + 1.0) / v.size())); s += b; }
      return s;
    };
    // envelope of the full epistemic box
    std::string s = cdf(vl); char b[64];
    for (size_t i = vh.size(); i-- > 0;) { std::snprintf(b, sizeof b, "L%.1f %.1f", X(vh[i]), a.Y((i + 1.0) / vh.size())); s += b; }
    std::fprintf(f, "<path d='%sZ' fill='rgba(231,156,52,0.30)' stroke='none'/>\n", s.c_str());
    std::fprintf(f, "<path d='%s' fill='none' stroke='#b8860b' stroke-width='1.5' stroke-dasharray='5 4'/>\n", cdf(fl).c_str());
    std::fprintf(f, "<path d='%s' fill='none' stroke='#b8860b' stroke-width='1.5' stroke-dasharray='5 4'/>\n", cdf(fh).c_str());
    std::fprintf(f, "<path d='%s' fill='none' stroke='#0b2a5c' stroke-width='2.2'/>\n", cdf(v0).c_str());
    std::fprintf(f, "<line x1='%.1f' x2='%.1f' y1='%g' y2='%g' stroke='#c0392b' stroke-width='1.5'/><text x='%.1f' y='%g' font-size='12' fill='#c0392b'>MIC50</text>\n", X(60), X(60), a.y0, a.y0 + a.hgt, X(60) + 4, a.y0 + 14);
    std::fprintf(f, "<text x='%g' y='%g' font-size='12' fill='#0b2a5c'>— paper’s values, IIV only</text>\n", a.x0 + a.w + 8, a.y0 + 40);
    std::fprintf(f, "<text x='%g' y='%g' font-size='12' fill='#b8860b'>- - fu_pls interval only</text>\n", a.x0 + a.w + 8, a.y0 + 64);
    std::fprintf(f, "<text x='%g' y='%g' font-size='12' fill='#8b6508'>▮ fu_pls × pH_LYS</text>\n", a.x0 + a.w + 8, a.y0 + 88);
    std::fprintf(f, "</svg>\n"); std::fclose(f);
  }
  return 0;
}

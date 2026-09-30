import Foundation
import Evidence

/// A cena da combustão de hidrogénio — contraparte de `web/src/components/H2.tsx`.
///
/// A tese é outra que a da dose. Lá: um ponto não cruza uma linha. Aqui: uma
/// banda que muda quando se muda o passo de integração está a reportar o
/// integrador, não a química.
///
/// O Swift calcula as três larguras medidas e escreve-as em `data-dt*`. A ilha
/// só troca qual está aplicada — não recalcula, porque o cálculo pertence a
/// onde há evidência.
enum H2 {
    static let id = "site/h2_scene.v1"

    static let species = ["H", "O", "OH", "H2O", "HO2", "H2O2", "H2", "O2"]
    /// Banda dominada pela incerteza da mistura inicial: não acumula.
    static let nonAccumulating: Set<String> = ["H2", "O2"]

    /// Três posições porque a medição tem três. Um contínuo insinuaria um
    /// contínuo que ninguém mediu.
    static let steps: [(key: String, label: String)] = [
        ("dt4", "4e-9 s"), ("dt2", "2e-9 s"), ("dt1", "1e-9 s"),
    ]

    /// Largura relativa em cada passo, com dt = 4e-9 valendo 1.
    static func widths(_ ev: Evidence, side: String, species s: String) throws -> [String: Double] {
        func v(_ m: String) throws -> Double { try ev.claim(id, metric: "\(side).\(s).\(m)").value }
        return ["dt4": 1, "dt2": 1 / (try v("half1")), "dt1": 1 / (try v("quarter"))]
    }

    static func f3(_ v: Double) -> String { String(format: "%.3f", v) }

    static func sweep(_ ev: Evidence) throws -> String {
        var rows = ""
        for s in species {
            let q = try widths(ev, side: "quadrature", species: s)
            let n = try widths(ev, side: "native", species: s)
            func data(_ w: [String: Double]) -> String {
                steps.map { " data-\($0.key)=\"\(f3(w[$0.key] ?? 0))\"" }.joined()
            }
            let flat = nonAccumulating.contains(s) ? " data-flat" : ""
            rows += """
            <li\(flat)>
              <span class="sp-name">\(Design.esc(s))</span>
              <span class="sp-track">
                <span class="sp-bar" data-side="quadrature"\(data(q))\
             style="width:\(f3((q["dt4"] ?? 0) * 100))%"></span>
                <span class="sp-bar" data-side="native"\(data(n))\
             style="width:\(f3((n["dt4"] ?? 0) * 100))%"></span>
              </span>
              <span class="sp-n" data-side="quadrature"\(data(q))>\(f3(q["dt4"] ?? 0))</span>
              <span class="sp-n" data-side="native"\(data(n))>\(f3(n["dt4"] ?? 0))</span>
            </li>
            """
        }

        let buttons = steps.enumerated().map { i, s in
            """
            <button type="button" class="step" data-step="\(s.key)" \
            aria-pressed="\(i == 0)">dt = \(s.label)</button>
            """
        }.joined()

        let foot = Pages.para([
            .text(try Prose("H2 and O2 do not move on either side. Their band is dominated by the uncertainty seeded in the initial mixture, which does not accumulate — so the rule that the band scales with the step is false for ")),
            .claim(try ev.claim(id, metric: "non_accumulating_count")),
            .text(try Prose(" of the ")),
            .claim(try ev.claim(id, metric: "species_reported")),
            .text(try Prose(" species reported. Saying it without that qualifier would be the same kind of overstatement this page is about.")),
        ], cls: "sweep-foot")

        return """
        <div class="sweep" data-island="h2-sweep" data-step="dt4">
          <div class="sweep-head">
            <p>1-σ band width, relative to its own value at the coarsest step</p>
            <div class="steps" role="group" aria-label="Integration step">\(buttons)</div>
          </div>
          <div class="sweep-legend">
            <span data-side="quadrature">per-step quadrature</span>
            <span data-side="native">Sounio, coherent propagation</span>
          </div>
          <ul class="sp">\(rows)</ul>
          \(foot)
        </div>

        <p class="thesis">Refining the step made one band shrink and left the other alone.<br>
        <em>A band that tracks your solver is reporting the solver.</em></p>
        """
    }

    /// O contraste em três implementações independentes, sob um fator quatro no passo.
    static func contrast(_ ev: Evidence) throws -> String {
        let rows: [(String, String, String)] = [
            ("replica", "Python replica", "per-step independent quadrature"),
            ("cpp", "C++ cross-check", "same formula, independently written from the published protocol rather than translated"),
            ("native", "Sounio native", "coherent sensitivity propagation"),
        ]
        let out = try rows.map { k, name, note in
            """
            <div class="cr"\(k == "native" ? " data-native" : "")>
              <span class="cr-name">\(Design.esc(name))</span>
              <span class="cr-val">\(Pages.chip(try ev.claim(id, metric: "\(k)_lo"))) – \
            \(Pages.chip(try ev.claim(id, metric: "\(k)_hi")))</span>
              <span class="cr-note">\(Design.esc(note))</span>
            </div>
            """
        }.joined()
        return "<div class=\"contrast\">\(out)</div>"
    }
}

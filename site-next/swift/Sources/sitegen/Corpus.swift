import Foundation
import Evidence

/// O corpus escrito, desenhado — a contraparte de `web/src/components/Corpus.tsx`.
///
/// Os dois geradores leem o MESMO artefato, incluindo a escala do campo
/// (`mark_lines`). Se a escala vivesse no código de cada um, eles poderiam
/// desenhar o mesmo corpus em tamanhos diferentes e nada acusaria.
enum Corpus {
    static let id = "site/corpus.v1"

    /// A ordem é semântica, não decrescente: primeiro o que sustenta a tese,
    /// depois o material de apoio. Ordenar por tamanho poria os testes à frente
    /// do Lean e desfaria o argumento.
    static let domains: [(key: String, label: String)] = [
        ("compiler", "the compiler, written in Sounio"),
        ("stdlib",   "standard library"),
        ("lean",     "machine-checked semantics, Lean 4"),
        ("examples", "examples and benchmarks"),
        ("tests",    "test corpus"),
        ("tooling",  "tooling and experiments"),
        ("docs",     "documentation programs"),
    ]

    static let trees: [(key: String, label: String)] = [
        ("artifacts", "generated scale probes"),
        ("archive",   "superseded bootstrap images"),
        ("bootstrap", "the generated bootstrap image"),
    ]

    struct Row { let key: String, label: String; let lines: Int, files: Int, marks: Int }

    /// Lança se o artefato não existir: sem medição, não há abertura.
    static func rows(_ ev: Evidence) throws -> [Row] {
        let perMark = try ev.claim(id, metric: "mark_lines").value
        return try domains.compactMap { d in
            guard let lines = try? ev.claim(id, metric: "\(d.key).lines") else { return nil }
            let files = try ev.claim(id, metric: "\(d.key).files")
            return Row(key: d.key, label: d.label,
                       lines: Int(lines.value), files: Int(files.value),
                       marks: Int((lines.value / perMark).rounded()))
        }
    }

    static func n(_ v: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.locale = Locale(identifier: "en_GB")
        return f.string(from: NSNumber(value: v)) ?? String(v)
    }

    /// O campo: uma marca por fatia igual de linhas, dimensionado por LINHA.
    static func field(_ ev: Evidence) throws -> String {
        let rs = try rows(ev)
        let perMark = Int(try ev.claim(id, metric: "mark_lines").value)
        let written = Int(try ev.claim(id, metric: "written.lines").value)
        let total = rs.reduce(0) { $0 + $1.marks }

        let marks = rs.map { r in
            "<span class=\"field-run\" data-d=\"\(r.key)\">"
            + String(repeating: "<i data-d=\"\(r.key)\"></i>", count: r.marks)
            + "</span>"
        }.joined()

        let legend = rs.map { r in
            """
            <li data-d="\(r.key)"><b>\(n(r.lines))</b>\
            <span>\(Design.esc(r.label))</span><em>\(n(r.files)) files</em></li>
            """
        }.joined()

        return """
        <div class="field" role="img" aria-label="The written corpus: \(n(written)) lines across \(rs.count) domains.">\(marks)</div>
        <p class="field-cap">\(n(total)) marks · 1 mark = \(n(perMark)) written lines · grouped by what they are</p>
        <ul class="legend">\(legend)</ul>
        """
    }

    /// A mesma medição achatada, para páginas onde o campo é demais.
    static func bar(_ ev: Evidence) throws -> String {
        let rs = try rows(ev)
        let written = Int(try ev.claim(id, metric: "written.lines").value)
        let segs = rs.map {
            "<span data-d=\"\($0.key)\" style=\"flex:\($0.lines)\" title=\"\(Design.esc($0.label)): \(n($0.lines)) lines\"></span>"
        }.joined()
        let key = rs.map { "<li data-d=\"\($0.key)\"><b>\(n($0.lines))</b> \($0.key)</li>" }.joined()
        return """
        <div class="cbar" role="img" aria-label="Written corpus by domain, \(n(written)) lines in total.">\(segs)</div>
        <ul class="ckey">\(key)</ul>
        """
    }

    /// O que ficou de fora, na mesma escala.
    static func excluded(_ ev: Evidence) throws -> String {
        let ts: [(key: String, label: String, lines: Int)] = trees.compactMap { t in
            guard let c = try? ev.claim(id, metric: "tree.\(t.key).lines") else { return nil }
            return (t.key, t.label, Int(c.value))
        }
        let segs = ts.map {
            "<span data-x=\"\($0.key)\" style=\"flex:\($0.lines)\" title=\"\($0.key): \(n($0.lines)) lines\"></span>"
        }.joined()
        let list = ts.map {
            "<li><b>\(n($0.lines))</b> \($0.key) — \(Design.esc($0.label))</li>"
        }.joined()
        return """
        <div class="strip" role="img" aria-label="Excluded trees, by size.">\(segs)</div>
        <ul class="excl">\(list)</ul>
        """
    }
}

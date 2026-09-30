import Foundation
import Evidence

/// As páginas, no registro de leitura de instrumento.
/// Toda prosa passa por `Prose`, todo numeral por `Claim` ou pelo índice.
enum Pages {

    // ---------------------------------------------------------------- peças

    static func chip(_ c: Claim) -> String {
        let level = c.level?.rawValue ?? "record"
        return """
        <span class="claim" data-level="\(level)" \
        title="\(Design.esc(c.artifact)) · \(Design.esc(c.metric))">\
        <span class="claim-glyph" aria-hidden="true">\(c.glyph)</span>\
        <span class="claim-value">\(c.formatted)</span></span>
        """
    }

    enum Part {
        case text(Prose), claim(Claim), code(String), strong(String)
    }

    static func para(_ parts: [Part], cls: String = "lede") -> String {
        let inner = parts.map { p -> String in
            switch p {
            case .text(let t):   return Design.esc(t.text)
            case .claim(let c):  return chip(c)
            case .code(let s):   return "<code>\(Design.esc(s))</code>"
            case .strong(let s): return "<b>\(Design.esc(s))</b>"
            }
        }.joined()
        return "<p class=\"\(cls)\">\(inner)</p>"
    }

    static func p(_ s: String, cls: String = "lede") throws -> String {
        para([.text(try Prose(s))], cls: cls)
    }

    static func readout(_ cells: [(String, String)]) -> String {
        let inner = cells.map { "<div><b>\($0.0)</b><span>\(Design.esc($0.1))</span></div>" }
            .joined()
        return "<div class=\"readout\">\(inner)</div>"
    }

    /// A banda: o que a amostra sustenta, desenhado.
    static func interval(_ ev: Evidence, _ a: Artifact, showId: Bool = true) -> String {
        guard let p = a.proportion else { return "" }
        let failed = a.level == .refused
        let t = String(format: "%.4f", ev.tension(p))
        let lo = String(format: "%.4f", p.lo * 100)
        let w = String(format: "%.4f", max(p.width * 100, 0.35))
        let pt = String(format: "%.4f", p.point * 100)
        let id = showId ? "<span class=\"iv-id\">\(Design.esc(a.id))</span>" : ""
        return """
        <div class="iv"\(failed ? " data-fail" : "") style="--t:\(t)">
          <span class="iv-n">\(p.passed)<i>/</i>\(p.total)</span>
          <span class="iv-track">
            <span class="iv-band" style="left:\(lo)%;width:\(w)%"></span>
            <span class="iv-point" style="left:\(pt)%"></span>
          </span>
          <span class="iv-text">[\(String(format: "%.3f", p.lo)), \(String(format: "%.3f", p.hi))]</span>
          \(id)
        </div>
        """
    }

    // ------------------------------------------------------------------ home

    static func home(_ ev: Evidence) throws -> String {
        let st = ev.stats
        let C = Corpus.id, D = Dose.id, R = Refusals.id

        return """
        <div class="wrap">
          <!-- 1. O MEDO — a cena é do leitor, não do projeto. -->
          <section class="band band-open" style="border-top:0">
            <p class="kicker">a dose calculation</p>
            <h1>Your analysis returned a number.<br><em>It was never that exact.</em></h1>
            \(para([
                .text(try Prose("A nurse weighs a patient: ")),
                .claim(try ev.claim(D, metric: "weight")),
                .text(try Prose(" kg. The guideline says ")),
                .claim(try ev.claim(D, metric: "per_kg")),
                .text(try Prose(" mg/kg. You multiply. Every language on earth gives you the same answer, instantly, with total confidence.")),
            ], cls: "lede lede-lg"))
            \(try Dose.panels(ev))
            \(try p("Same arithmetic. The left column discarded what you knew about your own measurement on line three, and nothing complained. That number went into a table; the table went into a paper.", cls: "note"))
          </section>

          <!-- 2. O GESTO — o argumento inteiro num movimento do dedo. -->
          <section class="band">
            <p class="kicker">the therapeutic window</p>
            <h2>The dose is safe between \(chip(try ev.claim(D, metric: "window_lo"))) and \(chip(try ev.claim(D, metric: "window_hi"))) mg.</h2>
            \(para([
                .claim(try ev.claim(D, metric: "dose")),
                .text(try Prose(" sits dead centre. It looks like the safest possible answer — and it will keep looking that way no matter how bad the scale is, because a point estimate has nowhere to move. Drag the scale's precision and watch what does. The window is a bound your program declares; what changes is that Sounio checks it against the interval instead of against the point.")),
            ], cls: "lede lede-lg"))
            \(try Dose.rig(ev))
          </section>

          <!-- 2b. A SEGUNDA CENA — outro campo, outra tese. -->
          <section class="band">
            <p class="kicker">hydrogen ignition · GRI-Mech 3.0 H/O</p>
            <h2>Refine your time step and your uncertainty shrinks. That is not convergence.</h2>
            \(para([
                .text(try Prose("You integrated a hydrogen mechanism — ")),
                .claim(try ev.claim(H2.id, metric: "species")),
                .text(try Prose(" species, ")),
                .claim(try ev.claim(H2.id, metric: "reactions")),
                .text(try Prose(" reactions — and reported a 1-σ band on the radicals. Halve the step and the band shrinks by about √2. Halve it again and it shrinks again. Nothing about the chemistry changed.")),
            ], cls: "lede lede-lg"))
            \(try H2.sweep(ev))
            \(try p("Three implementations, measured under a factor four in step size. Two of them were written independently of each other; they agree, and they are both wrong in the same way."))
            \(try H2.contrast(ev))
            \(para([
                .text(try Prose("The per-step form adds an independent uncertainty at every step, when successive steps share the same rate parameters and are not independent. Quadrature over dependent terms emits a bound tighter than the truth — which is the one direction of error this language treats as a lie rather than an approximation.")),
            ]))
            \(para([
                .text(try Prose("Sounio refuses that composition at compile time, and the refusal is in the corpus below as ")),
                .code("dsep_fork_unconditioned"),
                .text(try Prose(". Wiring the same refusal into the chemistry module's quadrature path is a separate, still-open change — so the correctness above is measured, not enforced, and this page says which is which.")),
            ], cls: "note"))
          </section>

          <!-- 3. A CREDENCIAL QUE IMPORTA — não o tamanho, a recusa. -->
          <section class="band">
            <p class="kicker">\(chip(try ev.claim(R, metric: "live"))) programs this compiler refuses</p>
            <h2>Not warnings. Not lint. It will not build.</h2>
            \(try p("Sounio ships a corpus of programs that must fail to compile. Each one is a way of being wrong that your current language compiles without a word. Three of them, unedited:", cls: "lede lede-lg"))
            \(try Refusals.list())
            \(para([
                .text(try Prose("The corpus holds ")),
                .claim(try ev.claim(R, metric: "live")),
                .text(try Prose(" live refusal tests across ")),
                .claim(try ev.claim(R, metric: "error_codes")),
                .text(try Prose(" declared error classes, plus ")),
                .claim(try ev.claim(R, metric: "ignored")),
                .text(try Prose(" currently disabled and marked as such in the source. The disabled ones are counted separately rather than folded into the headline, because a refusal that is switched off is not a refusal.")),
            ], cls: "note"))
          </section>

          <!-- 4. E NÃO É PROTÓTIPO — o tamanho entra depois do motivo. -->
          <section class="band">
            <h2>And it is not a prototype.</h2>
            \(para([
                .text(try Prose("The compiler that enforces the above is written in Sounio and compiles itself: ")),
                .claim(try ev.claim(C, metric: "compiler.lines")),
                .text(try Prose(" lines of it, inside a written corpus of ")),
                .claim(try ev.claim(C, metric: "written.lines")),
                .text(try Prose(" across ")),
                .claim(try ev.claim(C, metric: "written.files")),
                .text(try Prose(" files, with the semantics formalised in Lean — ")),
                .claim(try ev.claim(C, metric: "lean.lines")),
                .text(try Prose(" lines of it.")),
            ]))
            \(try Corpus.bar(ev))
            \(try p("Every numeral on this site, including all of the above, is read from artifacts at build time. A hand-written number fails the build. That is the same discipline the language applies to your data, applied to its own marketing."))
            \(readout([
                (String(st.artifacts), "artifacts indexed"),
                (String(st.gates), "gates with a verdict"),
                (String(st.refused), "gates that refuse"),
                (String(st.allPass), "reporting all-pass"),
            ]))
          </section>

          <!-- 5. A PORTA — a honestidade é o que faz o resto acreditável. -->
          <section class="band">
            <h2>Why you would believe any of this.</h2>
            \(try p("The flagship uncertainty example in this repository documents an error bar half the width of the one its own arithmetic produces. It was found while building this page, and instead of being fixed quietly it is written up on the honesty page, next to a formal model this project refuted and republished under a banner saying so.", cls: "lede lede-lg"))
            \(try p("That is worth more to you than any benchmark here. It is the one thing that tells you what happens in this project when something is found.", cls: "note"))
            <div class="cta">
              <a class="btn btn-solid" href="honesty.html">Read the argument</a>
              <a class="btn" href="proof.html">Browse the corpus</a>
            </div>
          </section>
        </div>
        """
    }

    // ----------------------------------------------------------------- proof

    static func proof(_ ev: Evidence) throws -> String {
        let s = ev.stats

        var blocks = ""
        for t in ev.territories {
            var rows = ""
            for a in t.artifacts {
                let mark = a.level.map { l -> String in
                    switch l { case .verified: "◈"; case .unbounded: "△"; case .refused: "⊘" }
                } ?? "·"
                let verdict = a.level.map { l -> String in
                    switch l {
                    case .verified: "passes"
                    case .unbounded: "no measured bound"
                    case .refused: "refused"
                    }
                } ?? "measurement · no verdict"
                let reason = a.reason.map { "<span class=\"row-reason\">\(Design.esc($0))</span>" } ?? ""
                let n = a.metrics.count
                let date = a.generatedAt.map { "<span>\(Design.esc(String($0.prefix(10))))</span>" } ?? ""
                let iv = a.proportion != nil
                    ? "<span class=\"row-iv\">\(interval(ev, a, showId: false))</span>" : ""

                rows += """
                <li class="row" data-kind="\(a.kind.rawValue)"\
                \(a.level.map { " data-level=\"\($0.rawValue)\"" } ?? "")>
                  <span class="row-mark">\(mark)</span>
                  <span class="row-id">\(Design.esc(a.id))</span>
                  <span class="row-verdict">\(verdict)</span>
                  <span class="row-meta">\(n > 0 ? "<span>\(n)m</span>" : "")\(date)</span>
                  \(reason)\(iv)
                </li>
                """
            }
            blocks += """
            <section class="territory">
              <header class="territory-head">
                <h3>\(Design.esc(t.name))</h3>
                <span class="territory-counts">
                  \(t.gates > 0 ? "<span>\(t.gates) gates</span>" : "")
                  \(t.records > 0 ? "<span>\(t.records) records</span>" : "")
                  \(t.refused > 0 ? "<span class=\"is-refused\">\(t.refused) refused</span>" : "")
                </span>
              </header>
              <ul class="rows">\(rows)</ul>
            </section>
            """
        }

        return """
        <div class="wrap">
          <section class="band" style="border-top:0">
            <p class="kicker">the corpus · \(s.artifacts) artifacts · unedited</p>
            <h1>What can be verified,<br><em>and what merely happened.</em></h1>
            \(try p("Every claim on this site resolves to a file in artifacts. The corpus holds two different things, and conflating them would be the first lie."))
            \(para([
                .text(try Prose("They were produced by a body of code this size — ")),
                .claim(try ev.claim(Corpus.id, metric: "written.lines")),
                .text(try Prose(" written lines across ")),
                .claim(try ev.claim(Corpus.id, metric: "written.files")),
                .text(try Prose(" files, of which ")),
                .claim(try ev.claim(Corpus.id, metric: "compiler.lines")),
                .text(try Prose(" are the compiler itself. That measurement is in this corpus too, as a record:")),
            ]))
            \(try Corpus.field(ev))
            \(para([
                .strong("Gates"),
                .text(try Prose(" declare a verdict — it passes, it fails, or its bound was never measured. ")),
                .strong("Records"),
                .text(try Prose(" carry measurement, research result and provenance. They have no verdict because they are not gates: a seed-refresh receipt has nothing to pass or fail.")),
            ]))
            \(readout([
                (String(s.gates), "gates"), (String(s.records), "records"),
                (String(s.refused), "refusals"), (String(s.proportions), "with an interval"),
                (String(format: "%.3f", s.widest), "widest interval"),
                (String(format: "%.3f", s.narrowest), "narrowest"),
            ]))
            <div class="controls">
              <div data-island="proof-filter" data-target=".territories"></div>
              \(para([
                  .text(try Prose("Grouped by territory. Within each, refusals first. Where a gate reports passed over total, its Wilson 95% interval is drawn. All ")),
                  .strong(String(s.artifacts)),
                  .text(try Prose(" rows are in this HTML — the filter only hides them.")),
              ], cls: "note"))
            </div>
          </section>
          <div class="territories">\(blocks)</div>
        </div>
        """
    }

    // --------------------------------------------------------------- honesty

    private struct Act {
        let when: String, title: String, teaches: String, body: [String]
    }

    static func honesty(_ ev: Evidence) throws -> String {
        let s = ev.stats
        let ratchet = "gates/fn_type_effect_ratchet"
        let census = "research/cross_engine_runpass_census/metrics_check_latest"

        let acts = [
            Act(when: "08:00 · the model",
                title: "A theorem can be false about your own language.",
                teaches: "A formal model is a claim about a system, and the system can refute it. When they disagree, one of them is wrong — and it is not automatically the implementation.",
                body: [
                    "EpistemicEffects.lean lost subject reduction. A preservation counterexample was proved, and the ruling was recorded: collapsing a measurement payload to Real is a defect in the model, not a design choice in the language. The checker had been right all along — measure of a Nat is Knowledge of a Nat.",
                    "The consequence is in the repository and not softened: do not cite that file as metatheory. It is a refuted model, and it stays published under a banner saying so. Deleting it would have been the dishonest option.",
                ]),
            Act(when: "14:00 · the instrument",
                title: "A proof is more useful as a lamp than as a trophy.",
                teaches: "Once refusal can be stated formally, you can go looking for places that fail to refuse. The theorem stops being a credential and becomes a search procedure.",
                body: [
                    "Two properties were proved in the E219 fragment with no sorry: that refusal is not zero, and that a well-typed program yields a value or refuses. That file is a model. It does not prove the checker implements the model — the gate stops at E4 and says so, which is the difference between this page and marketing.",
                    "The same afternoon the model was turned against the compiler and found two sites still fabricating after a refusal. An absolute value plus one was still typed as an integer where it should have been an error type. An empty IR stub still read as a fall-through zero; it now emits a trap.",
                ]),
            Act(when: "01:00 · the same absence",
                title: "Four lanes, one bug class: a plausible value where absence belonged.",
                teaches: "The interesting defects are not wrong numbers. They are plausible numbers standing in for something never measured — invisible until you have a word for them.",
                body: [
                    "Four lanes sharing no write-set landed the same discovery on four surfaces. A pin counter read as zero live pins when the emitter never incremented it; its sentinel now means unwired rather than empty. A call instruction was refused only as an unknown opcode; it now has a name and declares that it has no value. And a reclamation wall shipped its own failing witness before the fix existed.",
                    "The class is not closed — one surface is still open on the dissertation. What changed that night is that the class became visible, and visible is the prerequisite for closed.",
                ]),
        ]

        var actsHTML = ""
        for a in acts {
            let body = try a.body.map { try p($0) }.joined(separator: "\n")
            actsHTML += """
            <li class="act">
              <p class="act-when">\(Design.esc(a.when))</p>
              <div class="act-body">
                <h2>\(Design.esc(a.title))</h2>
                \(try p(a.teaches, cls: "act-teaches"))
                \(body)
              </div>
            </li>
            """
        }

        let ratchetBand = ev.artifacts[ratchet].map { interval(ev, $0, showId: false) } ?? ""

        return """
        <div class="wrap">
          <section class="band" style="border-top:0">
            <p class="kicker">17–18 august 2026 · one day</p>
            <h1>The model was wrong. The compiler was not.<br><em>The silent zero was the lie.</em></h1>
            \(try p("This page is one day, not a catalogue of controls. In twenty-four hours a formal model lost a theorem, a proof was turned into a lamp and found two places still fabricating, and four independent lanes named the same absence on four surfaces."))
            \(para([
                .text(try Prose("Everything else in this language — the types, the gates, the compiler — exists to make one sentence enforceable: ")),
                .strong("a well-typed program returns a value, or it refuses."),
            ]))
          </section>

          <ol class="acts">\(actsHTML)</ol>

          <section class="band">
            <h2>The same discipline, turned on this page</h2>
            \(para([
                .text(try Prose("The third act names a bug class: ")),
                .strong("a plausible value standing in for something that was never what it claimed"),
                .text(try Prose(". While measuring the corpus for the opening of this site, I produced one.")),
            ]))
            \(para([
                .text(try Prose("An early pass weighted the field by file and swept everything unclassified into a bucket labelled ")),
                .strong("tooling"),
                .text(try Prose(". That bucket held ")),
                .claim(try ev.claim(Corpus.id, metric: "excluded.lines")),
                .text(try Prose(" lines across ")),
                .claim(try ev.claim(Corpus.id, metric: "excluded.files")),
                .text(try Prose(" files — an average thick enough to give it away. Almost none of it was written. It is generated scale probes, the largest a single file of ")),
                .claim(try ev.claim(Corpus.id, metric: "biggest_generated")),
                .text(try Prose(" lines, and superseded bootstrap images kept for the record.")),
            ]))
            \(para([
                .text(try Prose("Counted as written, the opening figure would have been ")),
                .claim(try ev.claim(Corpus.id, metric: "total_tracked")),
                .text(try Prose(" rather than ")),
                .claim(try ev.claim(Corpus.id, metric: "written.lines")),
                .text(try Prose(". Three trees are excluded from every figure on this site, and the exclusion is published at the same scale as the thing it removes:")),
            ]))
            \(try Corpus.excluded(ev))
            \(para([
                .text(try Prose("Counting generated output as authored work is the first thing a project like this has to refuse, and the refusal is worth nothing unless the amount refused is stated. The written corpus is ")),
                .claim(try ev.claim(Corpus.id, metric: "written.lines")),
                .text(try Prose(" lines; what you see above is what that number would have been inflated by.")),
            ], cls: "note"))
          </section>

          <section class="band">
            <h2>The example on the front page understated its own uncertainty</h2>
            \(try p("The clinical scene that opens this site is not a mock-up. Its numbers are extracted at build time from the flagship demonstration of GUM uncertainty propagation in this repository.", cls: "lede lede-lg"))
            \(para([
                .text(try Prose("Its header documents the expected output as σ = ")),
                .claim(try ev.claim(Dose.id, metric: "documented_sd")),
                .text(try Prose(" mg, interval [")),
                .claim(try ev.claim(Dose.id, metric: "documented_ci_lo")),
                .text(try Prose(", ")),
                .claim(try ev.claim(Dose.id, metric: "documented_ci_hi")),
                .text(try Prose("]. The arithmetic the file itself writes out gives a variance of ")),
                .claim(try ev.claim(Dose.id, metric: "variance")),
                .text(try Prose(", so σ = ")),
                .claim(try ev.claim(Dose.id, metric: "sd")),
                .text(try Prose(" mg and interval [")),
                .claim(try ev.claim(Dose.id, metric: "ci_lo")),
                .text(try Prose(", ")),
                .claim(try ev.claim(Dose.id, metric: "ci_hi")),
                .text(try Prose("] — a band ")),
                .claim(try ev.claim(Dose.id, metric: "computed_width")),
                .text(try Prose(" mg wide where the header promises ")),
                .claim(try ev.claim(Dose.id, metric: "documented_width")),
                .text(try Prose(".")),
            ]))
            \(try p("Which of the two is wrong is not yet settled, and this page will not pretend otherwise. The header may simply be stale. Or the code may have gained an uncertainty on the dosing guideline that the header never intended — in which case the question is whether a clinical guideline range belongs in the propagation at all. That is a modelling decision, not a typo, and it is open."))
            \(try p("What is settled is that they disagree, and that the disagreement runs in the direction this language exists to prevent: the documented figure claims to know more than the computation supports. The build now extracts these parameters from the example rather than restating them, so the scene on the front page cannot drift from the file it illustrates — which is how the disagreement surfaced in the first place.", cls: "note"))
          </section>

          <section class="band">
            <h2>Where the argument is enforced</h2>
            \(try p("Not decoration. Three places you can check, with the numbers read from the corpus."))
            <div class="enforce">
              <div>
                <h3>Types</h3>
                \(para([
                    .text(try Prose("Measurement carries its payload type; a refusal is an error type, not a zero. The effect ratchet passes ")),
                    .claim(try ev.claim(ratchet, metric: "passed")),
                    .text(try Prose(" of ")),
                    .claim(try ev.claim(ratchet, metric: "total")),
                    .text(try Prose(" — which is this band, not certainty:")),
                ]))
                \(ratchetBand)
              </div>
              <div>
                <h3>Two engines</h3>
                \(para([
                    .text(try Prose("Madaros and Lean over ")),
                    .claim(try ev.claim(census, metric: "total")),
                    .text(try Prose(" cases, diverging explicitly on ")),
                    .claim(try ev.claim(census, metric: "diverge_explicit")),
                    .text(try Prose(". The divergences are published, because a disagreement between your own engines is evidence.")),
                ]))
              </div>
              <div>
                <h3>The ladder</h3>
                \(para([
                    .text(try Prose("Every claim stops at a level, E0 through E5, stated rather than implied. Stopping at E4 is not a failure. Saying you reached E5 when you stopped at E4 is. Across the corpus ")),
                    .strong(String(s.allPass)),
                    .text(try Prose(" gates report all-pass, and not one of them reaches certainty.")),
                ]))
              </div>
            </div>
            \(try p("This page is the day, not a catalogue of controls. The instruments appear where the story needs them — and the story is the argument.", cls: "note"))
            <div class="cta"><a class="btn btn-solid" href="proof.html">See the corpus</a></div>
          </section>
        </div>
        """
    }
}

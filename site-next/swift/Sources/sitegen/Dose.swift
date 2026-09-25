import Foundation
import Evidence

/// A cena clínica e o instrumento — contraparte de `web/src/components/Dose.tsx`.
///
/// O Swift escreve o rig PARADO, no σ que a cena traz, e pendura os parâmetros
/// em `data-*`. A ilha lê esses atributos e usa `propagate` de `web/src/dose.ts`
/// — a mesma função que o React usa. Duas renderizações, um cálculo.
///
/// Sem JavaScript a página fica completa: o rig mostra a leitura no σ do
/// exemplo, que é um valor legítimo da cena e não um estado de erro.
enum Dose {
    static let id = "site/dose_scene.v1"

    struct Scene {
        let weight, perKg, perKgVar, windowLo, windowHi, z: Double
        let axisLo, axisHi, tickStep: Double
        let weightSd, perKgSd, sigMin, sigMax, sigStep: Double
    }

    static func scene(_ ev: Evidence) throws -> Scene {
        func v(_ m: String) throws -> Double { try ev.claim(id, metric: m).value }
        return Scene(
            weight: try v("weight"), perKg: try v("per_kg"), perKgVar: try v("per_kg_var"),
            windowLo: try v("window_lo"), windowHi: try v("window_hi"), z: try v("z"),
            axisLo: try v("axis_lo"), axisHi: try v("axis_hi"), tickStep: try v("tick_step"),
            weightSd: try v("weight_sd"), perKgSd: try v("per_kg_sd"),
            sigMin: try v("sigma_min"), sigMax: try v("sigma_max"), sigStep: try v("sigma_step"))
    }

    struct Reading { let mean, sd, lo, hi, variance: Double; let outside: Bool }

    /// Método delta: Var(XY) = Y²Var(X) + X²Var(Y) — a fórmula que o exemplo escreve.
    static func propagate(_ s: Scene, weightSd: Double) -> Reading {
        let variance = s.perKg * s.perKg * weightSd * weightSd
                     + s.weight * s.weight * s.perKgVar
        let sd = variance.squareRoot()
        let mean = s.weight * s.perKg
        let lo = mean - s.z * sd, hi = mean + s.z * sd
        return Reading(mean: mean, sd: sd, lo: lo, hi: hi, variance: variance,
                       outside: lo < s.windowLo || hi > s.windowHi)
    }

    static func scaleKind(_ sd: Double) -> String {
        if sd <= 0.8 { return "calibrated clinical scale" }
        if sd <= 2.5 { return "ward scale" }
        if sd <= 4.5 { return "old scale, patient in clothes" }
        return "eyeballed from the chart"
    }

    static func f1(_ v: Double) -> String { String(format: "%.1f", v) }
    static func pos(_ s: Scene, _ mg: Double) -> Double {
        (mg - s.axisLo) / (s.axisHi - s.axisLo) * 100
    }
    static func pc(_ v: Double) -> String { String(format: "%.4f%%", v) }

    // ----------------------------------------------------------- os painéis

    static func panels(_ ev: Evidence) throws -> String {
        let s = try scene(ev)
        let r = propagate(s, weightSd: s.weightSd)
        let left = """
        weight = \(f1(s.weight))
        per_kg = \(f1(s.perKg))
        dose   = weight * per_kg
        print(dose)
        """
        let right = """
        let weight: Knowledge<f64> = measure(\(f1(s.weight)), uncertainty: \(f1(s.weightSd)))
        let per_kg: Knowledge<f64> = measure(\(f1(s.perKg)), uncertainty: \(f1(s.perKgSd)))
        let dose = weight * per_kg
        """
        return """
        <div class="scene">
          <div>
            <h3>Any language you already use</h3>
            <pre>\(Design.esc(left))</pre>
            <div class="out"><span>stdout</span><b>\(f1(r.mean))</b></div>
          </div>
          <div>
            <h3>Sounio</h3>
            <pre>\(Design.esc(right))</pre>
            <div class="out"><span>dose : Knowledge&lt;f64&gt;</span>\
        <b class="band">\(f1(r.mean)) ± \(f1(r.sd)) mg</b></div>
          </div>
        </div>
        """
    }

    // -------------------------------------------------------- o instrumento

    static func rig(_ ev: Evidence) throws -> String {
        let s = try scene(ev)
        let r = propagate(s, weightSd: s.weightSd)

        var ticks = ""
        var t = s.axisLo
        while t <= s.axisHi + 1e-9 {
            ticks += "<span style=\"left:\(pc(pos(s, t)))\">\(Corpus.n(Int(t.rounded())))</span>"
            t += s.tickStep
        }

        let refused = r.outside ? " data-refused" : ""
        let verdictValue = r.outside ? "REFUSED" : "\(f1(r.mean)) ± \(f1(r.sd)) mg"
        let verdictWhy = r.outside
            ? "The interval [\(f1(r.lo)), \(f1(r.hi))] leaves the window. The bound is "
              + "checked against the interval, so there is no longer a safe dose to assert."
            : "Interval [\(f1(r.lo)), \(f1(r.hi))] — inside the window"

        return """
        <div class="rig" data-island="dose-rig"\
         data-weight="\(s.weight)" data-per-kg="\(s.perKg)" data-per-kg-var="\(s.perKgVar)"\
         data-window-lo="\(s.windowLo)" data-window-hi="\(s.windowHi)" data-z="\(s.z)"\
         data-axis-lo="\(s.axisLo)" data-axis-hi="\(s.axisHi)" data-tick-step="\(s.tickStep)">
          <div class="rig-head">
            <p>dose = weight × \(f1(s.perKg)) mg/kg · GUM propagation · 95% coverage</p>
            <p class="rig-var">Var = \(Int(r.variance.rounded()))</p>
          </div>

          <div class="axis"\(refused)>
            <div class="win" style="left:\(pc(pos(s, s.windowLo)));\
        width:\(pc(pos(s, s.windowHi) - pos(s, s.windowLo)))"></div>
            <span class="win-lab" style="left:\(pc(pos(s, s.windowLo)))">\(Corpus.n(Int(s.windowLo))) mg</span>
            <span class="win-lab" style="left:\(pc(pos(s, s.windowHi)))">\(Corpus.n(Int(s.windowHi))) mg</span>
            <div class="edge" style="left:\(pc(pos(s, s.windowLo)))"></div>
            <div class="edge" style="left:\(pc(pos(s, s.windowHi)))"></div>
            <div class="bandbar" style="left:\(pc(pos(s, r.lo)));\
        width:\(pc(pos(s, r.hi) - pos(s, r.lo)))"></div>
            <div class="pointmk" style="left:\(pc(pos(s, r.mean)))"></div>
            <span class="pointlab" style="left:\(pc(pos(s, r.mean)))">\(f1(r.mean))</span>
            <div class="ticks">\(ticks)</div>
          </div>

          <div class="ctl">
            <div>
              <label for="dose-sd">how well the patient was weighed</label>
              <input id="dose-sd" type="range" min="\(s.sigMin)" max="\(s.sigMax)"\
         step="\(s.sigStep)" value="\(s.weightSd)">
              <div class="ctl-val">
                <span>σ = <b>\(f1(s.weightSd))</b> kg</span>
                <span>\(scaleKind(s.weightSd))</span>
              </div>
            </div>
            <div class="verdict"\(refused)>
              <div class="verdict-k">what can still be asserted</div>
              <div class="verdict-v">\(verdictValue)</div>
              <div class="verdict-w">\(Design.esc(verdictWhy))</div>
            </div>
          </div>
        </div>

        <p class="thesis">The point estimate never moved off \(f1(r.mean)).<br>
        <em>A point cannot cross a line. Only a band can.</em></p>
        """
    }
}

/// As recusas — contraparte de `web/src/components/Refusals.tsx`.
enum Refusals {
    static let id = "site/refusals.v1"

    /// Nomes reais de `tests/compile-fail/`, sem edição. O `what` é montado em
    /// partes porque os pesos são LITERAIS CITADOS do arquivo de teste: passam
    /// como código, não como afirmação — o mesmo caminho que o React usa, e o
    /// mesmo que o portão de prosa reconhece.
    static func shown() throws -> [(id: String, what: [Pages.Part], why: Prose)] {
        [
            ("dsep_fork_unconditioned",
             [.text(try Prose("Two quantities composed in quadrature as if independent, when a common cause makes them dependent."))],
             try Prose("The test's own comment states the consequence: it would emit a bound tighter than the truth. An error bar too narrow is the one kind this language will not let you write.")),

            ("audit_posterior_weights_not_normalized",
             [.text(try Prose("Bayesian model averaging where the posterior weights are ")),
              .code("0.6"), .text(try Prose(" and ")), .code("0.6"),
              .text(try Prose(". Posterior weights must sum to one."))],
             try Prose("Every statistical language you know will average with those weights and return a number. It will look completely normal.")),

            ("affine_double_use",
             [.text(try Prose("A value that may be consumed once, consumed twice."))],
             try Prose("The second read is of something that is no longer there. In C it is a use-after-move; in a data pipeline it is a sample counted twice.")),

            ("annotation_proof_mismatch",
             [.text(try Prose("A function annotated as carrying a proof, whose proof does not match the claim it is attached to."))],
             try Prose("The annotation is the thing a reader trusts. Letting it drift from what was actually proved is how a citation becomes a lie.")),
        ]
    }

    static func list() throws -> String {
        let rows = try shown().map { r in
            """
            <div class="ref">
              <span class="ref-id">\(Design.esc(r.id))</span>
              <div>\(Pages.para(r.what, cls: ""))\(Pages.para([.text(r.why)], cls: ""))</div>
            </div>
            """
        }.joined()
        return "<div class=\"refusals\">\(rows)</div>"
    }
}

import{j as n}from"./jsx-runtime.D_zvdyIk.js";import{r as a}from"./index.DYrVU9rO.js";const w="/wasm/sounio_compiler.js",b="/wasm/sounio_compiler_bg.wasm",u=[{name:"Hello World",code:`fn main() -> i32 with IO {
    print("Hello, World!")
    0
}`},{name:"Effects",code:`// Effects must be declared in the function signature.
// IO covers print/read; Mut covers mutable state.
fn greet(name: &str) -> i32 with IO {
    print("Hello, ")
    print(name)
    print("!")
    0
}

fn main() -> i32 with IO {
    greet("Sounio")
    0
}`},{name:"Type Error",code:`// This program has a deliberate type error.
// The checker will reject it: "not a number" is not i32.
fn main() -> i32 {
    let x: i32 = "not a number"
    x
}`},{name:"Epistemic / Knowledge",code:`// Knowledge<T> wraps a value with an uncertainty bound.
// measure() is the entry point into the epistemic chain.
fn main() -> i32 with IO {
    let dose: Knowledge<f64> = measure(500.0, uncertainty: 2.5)
    print("Measured dose with uncertainty tracking")
    0
}`},{name:"Measurement + Uncertainty",code:`// Dimensional units prevent unit-mismatch bugs at compile time.
// mg is a unit type; the compiler rejects mg + kg without conversion.
fn administer(patient_weight: kg, dose_per_kg: mg) -> mg with IO {
    let total: mg = patient_weight * dose_per_kg
    let k: Knowledge<mg> = measure(total, uncertainty: 1.5)
    print("Dose computed with uncertainty")
    total
}

fn main() -> i32 with IO {
    let w: kg  = 70.0
    let d: mg  = 15.0
    administer(w, d)
    0
}`},{name:"lift_knowledge",code:`// lift_knowledge promotes Knowledge<T> → Contest<T>.
// A Contest value is open to challenge: other agents may
// submit counterexamples or audit evidence.
fn main() -> i32 with IO {
    let k: Knowledge<f64> = measure(3.14159, uncertainty: 0.00001)
    let c: Contest<f64>   = lift_knowledge(k)
    print("Value is now contestable")
    0
}`},{name:"prove_robust",code:`// prove_robust(contest, proof) attaches a proof obligation.
// The proof is discharged at type-check time; zero runtime cost.
fn main() -> i32 with IO {
    let k: Knowledge<f64> = measure(9.81, uncertainty: 0.01)
    let c: Contest<f64>   = lift_knowledge(k)
    let r: Robust<f64>    = prove_robust(c, proof: "within_tolerance_0.1")
    print("Value is robust")
    0
}`},{name:"validate_manifest",code:`// validate_manifest discharges all pending proof obligations.
// The result is Validated<T>: audit-ready, no open proofs.
fn main() -> i32 with IO {
    let k: Knowledge<f64> = measure(9.81, uncertainty: 0.01)
    let c: Contest<f64>   = lift_knowledge(k)
    let r: Robust<f64>    = prove_robust(c, proof: "gravity_validated")
    let v: Validated<f64> = validate_manifest(r)
    print("Full epistemic chain complete")
    0
}`},{name:"Row-poly Effects",code:`// Row-polymorphic effects: a function that works with any
// effect row that includes IO. The <E> is an effect variable.
fn log_value<T, E>(val: T) -> () with IO | E {
    print("value logged")
}

fn pipeline() -> i32 with IO, Mut {
    log_value::<i32, Mut>(42)
    0
}

fn main() -> i32 with IO, Mut {
    pipeline()
}`},{name:"Aleatoric vs Epistemic",code:`// Aleatoric uncertainty: irreducible randomness (e.g. quantum noise).
// Epistemic uncertainty: reducible with more data (e.g. measurement error).
// The type system tracks which kind of uncertainty a value carries.
fn main() -> i32 with IO {
    // Aleatoric — cannot be reduced by gathering more evidence
    let noise: Aleatoric<f64> = aleatoric(0.03)

    // Epistemic — can be tightened with more measurements
    let est: Epistemic<f64>   = epistemic(measure(1.23, uncertainty: 0.05))

    print("Uncertainty kinds are distinct types")
    0
}`},{name:"Causal Types",code:`// Causal DAG with epistemic edge uncertainty
// Intervention<T> and Counterfactual<T> are compiler-level types
fn build_dag() -> i32 with IO {
    var dag = causal_dag_new()
    dag = dag_add_node(dag, 0)  // treatment
    dag = dag_add_node(dag, 1)  // outcome
    dag = dag_add_edge(dag, 0, 1,
        ec_beta_new(8.0, 2.0), 0.5, 0.1)
    let intervened = do_intervention(dag, 1)
    print("Causal intervention applied")
    0
}

fn main() -> i32 with IO {
    build_dag()
}`},{name:"Graded Effects",code:`// GradedEffect<ε, E> fuses modal grading with algebraic effects.
// ε is the epistemic grade (0 = certain, 1 = maximally uncertain).
// Effects with different grades cannot be silently composed.
fn certain_io() -> i32 with GradedEffect<0.0, IO> {
    print("certain output")
    0
}

fn uncertain_io() -> i32 with GradedEffect<0.4, IO> {
    print("uncertain output")
    0
}

fn main() -> i32 with IO {
    certain_io()
    uncertain_io()
    0
}`},{name:"Session Types (preview)",code:`// Session types enforce communication protocols at compile time.
// This is a preview — full session type inference is on the roadmap.
// The protocol below says: send i32, receive f64, close.
type Protocol = Send<i32, Recv<f64, End>>

fn run_session(ch: Channel<Protocol>) -> f64 with IO, Async {
    send(ch, 42)
    let result: f64 = recv(ch)
    close(ch)
    result
}

fn main() -> i32 with IO {
    print("Session types preview")
    0
}`},{name:"PBPK Drug Dosing",code:`// Physiologically-based pharmacokinetic (PBPK) model.
// Units enforce mg/kg dosing; epistemic types track clinical uncertainty.
fn vancomycin_dose(
    weight: kg,
    creatinine: umol_per_L,
) -> Knowledge<mg> with IO {
    // Cockroft-Gault clearance estimate (simplified)
    let cl_factor: f64 = 1.0
    let base_dose: mg  = weight * 15.0
    let adjusted: mg   = base_dose * cl_factor
    measure(adjusted, uncertainty: adjusted * 0.12)
}

fn main() -> i32 with IO {
    let dose = vancomycin_dose(70.0, 88.0)
    print("Vancomycin dose computed with 12% uncertainty")
    0
}`},{name:"Fibonacci",code:`fn fib(n: i32) -> i32 {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() -> i32 with IO {
    let result = fib(10)
    print("fib(10) = ")
    0
}`}];function O(d=[]){return d.length===0?"":d.map(o=>{const r=o.line?` [${o.line}:${o.column??1}]`:"";return`${o.severity.toUpperCase()}${r}: ${o.message}`}).join(`
`)}function A({initialCode:d,theme:o="dark"}){const[r,m]=a.useState(d||u[0].code),[y,s]=a.useState(""),[x,g]=a.useState(!1),[v,_]=a.useState(u[0].name),[c,k]=a.useState(null),[f,h]=a.useState("loading");a.useEffect(()=>{const t=new URLSearchParams(window.location.search).get("code");if(t)try{m(atob(t))}catch{}},[]),a.useEffect(()=>{let e=!1;async function t(){try{const i=await import(w);if(await i.default(b),e)return;k({compile:i.compile,run:i.run,format:i.format,version:i.version}),h("ready")}catch(i){if(e)return;h("error"),s(`Failed to load Sounio WASM runtime.
Expected assets:
- ${w}
- ${b}

${String(i)}`)}}return t(),()=>{e=!0}},[]);const p=a.useCallback(async()=>{if(!c){s("WASM runtime is not ready. Build assets with: ../scripts/build_playground_wasm.sh");return}g(!0),s(`Compiling and running...
`);try{const e=c.run(r),t=JSON.parse(e),i=O(t.diagnostics);let l="";i&&(l+=`${i}

`),t.output&&t.output.length>0&&(l+=t.output),t.returnValue!=null&&(l+=`${l?`

`:""}Return: ${t.returnValue}`),l||(l=t.success?"Program finished successfully.":"Program failed with no output."),s(l)}catch(e){s(`Runtime error:
${String(e)}`)}finally{g(!1)}},[r,c]),S=a.useCallback(()=>{const e=btoa(r),t=new URL(window.location.href);t.searchParams.set("code",e),navigator.clipboard.writeText(t.toString()),s("Link copied to clipboard.")},[r]),E=a.useCallback(e=>{const t=u.find(i=>i.name===e);t&&(m(t.code),_(e),s(""))},[]),j=a.useCallback(e=>{e.key==="Enter"&&(e.ctrlKey||e.metaKey)&&(e.preventDefault(),p())},[p]),I=f==="ready"?"WASM ready":f==="loading"?"WASM loading":"WASM error";return n.jsxs("div",{className:`flex flex-col h-full ${o==="dark"?"bg-[#1e1e1e]":"bg-white"}`,children:[n.jsxs("div",{className:"flex items-center justify-between p-3 bg-[var(--color-navy-900)] border-b border-white/10",children:[n.jsxs("div",{className:"flex items-center gap-3",children:[n.jsx("button",{onClick:p,disabled:x||f!=="ready",className:"flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-green-500/50 text-white rounded-lg font-medium transition-colors",children:"Run"}),n.jsx("button",{onClick:S,className:"px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg font-medium transition-colors",children:"Share"}),n.jsx("select",{value:v,onChange:e=>E(e.target.value),className:"px-3 py-2 bg-white/10 text-white rounded-lg font-medium border-0 cursor-pointer",children:u.map(e=>n.jsx("option",{value:e.name,className:"bg-[var(--color-navy-900)]",children:e.name},e.name))})]}),n.jsx("span",{className:"text-white/60 text-sm",children:I})]}),n.jsxs("div",{className:"flex-1 grid grid-cols-1 lg:grid-cols-2 min-h-0",children:[n.jsxs("div",{className:"border-r border-white/10 flex flex-col",children:[n.jsx("div",{className:"p-2 text-white/60 text-xs font-mono border-b border-white/10",children:"main.sio"}),n.jsx("textarea",{value:r,onChange:e=>m(e.target.value),onKeyDown:j,className:`flex-1 p-4 font-mono text-sm leading-relaxed resize-none focus:outline-none ${o==="dark"?"bg-[#1e1e1e] text-[#d4d4d4]":"bg-white text-gray-800"}`,spellCheck:!1})]}),n.jsxs("div",{className:"flex flex-col",children:[n.jsx("div",{className:"p-2 text-white/60 text-xs font-mono border-b border-white/10",children:"Output"}),n.jsx("pre",{className:`flex-1 p-4 font-mono text-sm overflow-auto ${o==="dark"?"bg-[#1e1e1e] text-green-400":"bg-gray-50 text-gray-800"}`,children:y||'Click "Run" to execute your code...'})]})]}),n.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-[var(--color-navy-900)] text-white/60 text-xs font-mono border-t border-white/10",children:[n.jsx("span",{children:"Sounio v1.0.0-beta.5"}),n.jsx("span",{children:c?.version?c.version():"WASM"})]})]})}export{A as default};

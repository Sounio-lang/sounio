#!/usr/bin/env python3
import csv

DEGREY = "/workspace/sounio/examples/erdos/data/degrey"
OUT = "/workspace/sounio/examples/erdos/degrey_chi5.sio"
SCALE = 10**7
OX, OY = 5.0, 1.0

V = {}
with open(f"{DEGREY}/degrey_graph_vertices.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        V[int(row[0])] = (float(row[1]), float(row[2]))
NV = len(V)
E = []
with open(f"{DEGREY}/degrey_graph_edges.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        E.append((int(row[0]), int(row[1])))
NE = len(E)

CX = [round((V[i][0] + OX) * SCALE) for i in range(1, NV + 1)]
CY = [round((V[i][1] + OY) * SCALE) for i in range(1, NV + 1)]
flatE = []
for (u, w) in E:
    flatE.append(u); flatE.append(w)

TARGET = SCALE * SCALE   # 1.0 scaled
TOL = 10**11

CHUNK = 1500

def emit_init(name, vals, fn_prefix):
    """Emit chunked init functions that fill global array `name` via runtime
    assignment (global-var-literal-init indexing is broken in this compiler;
    [0;N] splat + runtime writes works)."""
    fns = []
    names = []
    n = len(vals)
    idx = 0
    fi = 0
    while idx < n:
        end = min(idx + CHUNK, n)
        lines = [f"fn {fn_prefix}{fi}() with Mut, Panic {{"]
        for k in range(idx, end):
            lines.append(f"    {name}[{k}] = {vals[k]}")
        lines.append("}")
        fns.append("\n".join(lines))
        names.append(f"{fn_prefix}{fi}")
        idx = end
        fi += 1
    return "\n".join(fns) + "\n", names

LOGIC = f"""
// ---- clause DB (originals + learned) ----
var nvars: i32 = 0
var nclauses: i32 = 0
var cstart: [i32; 1048576] = [0; 1048576]
var clen: [i32; 1048576] = [0; 1048576]
var clits: [i64; 8388608] = [0; 8388608]

// ---- assignment / trail ----
var assign: [i32; 6400] = [0; 6400]
var vlevel: [i32; 6400] = [0; 6400]
var reason: [i32; 6400] = [0; 6400]
var trail_lit: [i64; 6400] = [0; 6400]
var trail_len: i32 = 0
var trail_lim: [i32; 6400] = [0; 6400]
var cur_level: i32 = 0

var seen: [i32; 6400] = [0; 6400]
var LEARNT: [i64; 8192] = [0; 8192]
var llen: i32 = 0

var P_n: i32 = 0
var P_start: [i32; 1048576] = [0; 1048576]
var P_len: [i32; 1048576] = [0; 1048576]
var P_lits: [i64; 8388608] = [0; 8388608]

var n_conflicts: i64 = 0
var CONFLICT_CAP: i64 = 200000
var n_decisions: i64 = 0

fn print_digit(d: i64) with IO {{
    if d == 0 {{ print("0") }} else if d == 1 {{ print("1") }} else if d == 2 {{ print("2") }}
    else if d == 3 {{ print("3") }} else if d == 4 {{ print("4") }} else if d == 5 {{ print("5") }}
    else if d == 6 {{ print("6") }} else if d == 7 {{ print("7") }} else if d == 8 {{ print("8") }}
    else {{ print("9") }}
}}
fn print_dec(n: i64) with IO, Mut, Div, Panic {{
    if n == 0 {{ print("0") return }}
    var x = n
    if x < 0 {{ print("-") x = 0 - x }}
    var buf: [i64; 24] = [0; 24]
    var len: i32 = 0
    while x > 0 {{ buf[len] = x % 10  x = x / 10  len = len + 1 }}
    var i: i32 = len - 1
    while i >= 0 {{ print_digit(buf[i])  i = i - 1 }}
}}

fn lit_var(l: i64) -> i32 with Panic {{
    if l < 0 {{ return (0 - l) as i32 }}
    return l as i32
}}

fn db_add(arr: &[i64; 8192], n: i32) -> i32 with Mut, Panic {{
    if nclauses >= 1048576 {{ return 0 - 1 }}
    let idx = nclauses
    var start: i32 = 0
    if idx > 0 {{ start = cstart[idx - 1] + clen[idx - 1] }}
    if (start as i64) + (n as i64) > 8388608 {{ return 0 - 1 }}
    cstart[idx] = start
    clen[idx] = n
    var i: i32 = 0
    while i < n {{
        clits[start + i] = arr[i]
        let v = lit_var(arr[i])
        if v > nvars {{ nvars = v }}
        i = i + 1
    }}
    nclauses = idx + 1
    return idx
}}

fn litval(l: i64) -> i32 with Mut, Panic {{
    let v = lit_var(l)
    var a = assign[v]
    if l < 0 {{ a = 0 - a }}
    return a
}}

fn enqueue(lit: i64, rs: i32) with Mut, Panic {{
    let v = lit_var(lit)
    if lit < 0 {{ assign[v] = 0 - 1 }} else {{ assign[v] = 1 }}
    vlevel[v] = cur_level
    reason[v] = rs
    trail_lit[trail_len] = lit
    trail_len = trail_len + 1
}}

fn propagate() -> i32 with Mut, Panic {{
    var changed: i32 = 1
    while changed == 1 {{
        changed = 0
        var c: i32 = 0
        while c < nclauses {{
            let st = cstart[c]
            let ln = clen[c]
            var sat: i32 = 0
            var unassigned: i32 = 0
            var last_lit: i64 = 0
            var k: i32 = 0
            while k < ln {{
                let l = clits[st + k]
                let val = litval(l)
                if val == 1 {{ sat = 1 }}
                if val == 0 {{ unassigned = unassigned + 1  last_lit = l }}
                k = k + 1
            }}
            if sat == 0 {{
                if unassigned == 0 {{ return c }}
                if unassigned == 1 {{ enqueue(last_lit, c)  changed = 1 }}
            }}
            c = c + 1
        }}
    }}
    return 0 - 1
}}

fn emit_proof_clause() with Mut, Panic {{
    let idx = P_n
    var start: i32 = 0
    if idx > 0 {{ start = P_start[idx - 1] + P_len[idx - 1] }}
    if (start as i64) + (llen as i64) > 8388608 {{ return }}
    P_start[idx] = start
    P_len[idx] = llen
    var i: i32 = 0
    while i < llen {{ P_lits[start + i] = LEARNT[i]  i = i + 1 }}
    P_n = idx + 1
}}

fn emit_empty() with Mut, Panic {{
    let idx = P_n
    var start: i32 = 0
    if idx > 0 {{ start = P_start[idx - 1] + P_len[idx - 1] }}
    P_start[idx] = start
    P_len[idx] = 0
    P_n = idx + 1
}}

fn analyze(confl: i32) -> i32 with Mut, Panic {{
    var pathC: i32 = 0
    llen = 1
    var p: i64 = 0
    var ci: i32 = confl
    var index: i32 = trail_len - 1
    var go: i32 = 1
    while go == 1 {{
        let st = cstart[ci]
        let ln = clen[ci]
        var k: i32 = 0
        while k < ln {{
            let q = clits[st + k]
            let vq = lit_var(q)
            var skip: i32 = 0
            if p != 0 {{ if vq == lit_var(p) {{ skip = 1 }} }}
            if skip == 0 {{
                if seen[vq] == 0 {{
                    if vlevel[vq] > 0 {{
                        seen[vq] = 1
                        if vlevel[vq] >= cur_level {{ pathC = pathC + 1 }}
                        else {{ LEARNT[llen] = q  llen = llen + 1 }}
                    }}
                }}
            }}
            k = k + 1
        }}
        while seen[lit_var(trail_lit[index])] == 0 {{ index = index - 1 }}
        p = trail_lit[index]
        seen[lit_var(p)] = 0
        pathC = pathC - 1
        if pathC <= 0 {{ go = 0 }}
        else {{ ci = reason[lit_var(p)]  index = index - 1 }}
    }}
    LEARNT[0] = 0 - p
    var btlevel: i32 = 0
    var j: i32 = 1
    while j < llen {{
        let lv = vlevel[lit_var(LEARNT[j])]
        if lv > btlevel {{ btlevel = lv }}
        j = j + 1
    }}
    j = 1
    while j < llen {{ seen[lit_var(LEARNT[j])] = 0  j = j + 1 }}
    emit_proof_clause()
    return btlevel
}}

fn backjump(btlevel: i32) with Mut, Panic {{
    let target = trail_lim[btlevel + 1]
    var i: i32 = trail_len - 1
    while i >= target {{
        let v = lit_var(trail_lit[i])
        assign[v] = 0
        reason[v] = 0 - 1
        vlevel[v] = 0
        i = i - 1
    }}
    trail_len = target
    cur_level = btlevel
}}

var search_ptr: i32 = 1
fn pick_unassigned() -> i32 with Mut, Panic {{
    var v: i32 = 1
    while v <= nvars {{
        if assign[v] == 0 {{ return v }}
        v = v + 1
    }}
    return 0
}}

fn solve_cdcl() -> i32 with IO, Mut, Div, Panic {{
    cur_level = 0
    trail_lim[0] = 0
    var loops: i32 = 1
    while loops == 1 {{
        let confl = propagate()
        if confl >= 0 {{
            n_conflicts = n_conflicts + 1
            if n_conflicts % 20000 == 0 {{
                print("  [progress] conflicts="); print_dec(n_conflicts)
                print(" lemmas="); print_dec(P_n as i64)
                print(" clauses="); print_dec(nclauses as i64); print("\\n")
            }}
            if n_conflicts > CONFLICT_CAP {{ return 2 }}
            if cur_level == 0 {{ emit_empty()  return 0 }}
            let bt = analyze(confl)
            let lidx = db_add(&LEARNT, llen)
            if lidx < 0 {{ return 3 }}
            backjump(bt)
            enqueue(LEARNT[0], lidx)
        }} else {{
            let x = pick_unassigned()
            if x == 0 {{ return 1 }}
            n_decisions = n_decisions + 1
            cur_level = cur_level + 1
            trail_lim[cur_level] = trail_len
            enqueue(x as i64, 0 - 1)
        }}
    }}
    return 2
}}

fn reset_assign_full() with Mut, Panic {{
    var i: i32 = 0
    while i <= nvars {{ assign[i] = 0  i = i + 1 }}
}}

fn rup_add_lemma(pst: i32, pln: i32) with Mut, Panic {{
    if nclauses >= 1048576 {{ return }}
    let idx = nclauses
    var start: i32 = 0
    if idx > 0 {{ start = cstart[idx - 1] + clen[idx - 1] }}
    if (start as i64) + (pln as i64) > 8388608 {{ return }}
    cstart[idx] = start
    clen[idx] = pln
    var i: i32 = 0
    while i < pln {{ clits[start + i] = P_lits[pst + i]  i = i + 1 }}
    nclauses = idx + 1
}}

fn propagate_noenq() -> i32 with Mut, Panic {{
    var changed: i32 = 1
    while changed == 1 {{
        changed = 0
        var c: i32 = 0
        while c < nclauses {{
            let st = cstart[c]
            let ln = clen[c]
            var sat: i32 = 0
            var un: i32 = 0
            var last: i64 = 0
            var k: i32 = 0
            while k < ln {{
                let l = clits[st + k]
                let v = lit_var(l)
                var a = assign[v]
                if l < 0 {{ a = 0 - a }}
                if a == 1 {{ sat = 1 }}
                if a == 0 {{ un = un + 1  last = l }}
                k = k + 1
            }}
            if sat == 0 {{
                if un == 0 {{ return 1 }}
                if un == 1 {{
                    let v = lit_var(last)
                    if last < 0 {{ assign[v] = 0 - 1 }} else {{ assign[v] = 1 }}
                    changed = 1
                }}
            }}
            c = c + 1
        }}
    }}
    return 0
}}

fn verify(orig_nc: i32) -> i32 with Mut, Panic {{
    nclauses = orig_nc
    var pi: i32 = 0
    while pi < P_n {{
        let pst = P_start[pi]
        let pln = P_len[pi]
        reset_assign_full()
        var taut: i32 = 0
        var k: i32 = 0
        while k < pln {{
            let l = P_lits[pst + k]
            let v = lit_var(l)
            var want: i32 = 0 - 1
            if l < 0 {{ want = 1 }}
            if assign[v] != 0 {{ if assign[v] != want {{ taut = 1 }} }}
            assign[v] = want
            k = k + 1
        }}
        var ok: i32 = 0
        if taut == 1 {{ ok = 1 }} else {{
            let cf = propagate_noenq()
            if cf == 1 {{ ok = 1 }}
        }}
        if ok == 0 {{ return 0 - 1 }}
        if pln == 0 {{ return 1 }}
        rup_add_lemma(pst, pln)
        pi = pi + 1
    }}
    return 0
}}

fn emit_dimacs(orig_nc: i32) with IO, Mut, Div, Panic {{
    print("p cnf ")  print_dec(nvars as i64)  print(" ")  print_dec(orig_nc as i64)  print("\\n")
    var c: i32 = 0
    while c < orig_nc {{
        let st = cstart[c]
        let ln = clen[c]
        var k: i32 = 0
        while k < ln {{ print_dec(clits[st + k])  print(" ")  k = k + 1 }}
        print("0\\n")
        c = c + 1
    }}
}}

fn emit_drat() with IO, Mut, Div, Panic {{
    var pi: i32 = 0
    while pi < P_n {{
        let pst = P_start[pi]
        let pln = P_len[pi]
        var k: i32 = 0
        while k < pln {{ print_dec(P_lits[pst + k])  print(" ")  k = k + 1 }}
        print("0\\n")
        pi = pi + 1
    }}
}}

// ---- de Grey graph data check (integer fixed-point, scale 1e7) ----
fn validate_geometry() -> i32 with IO, Mut, Panic {{
    var ok: i32 = 0
    var e: i32 = 0
    while e < {NE} {{
        let u = E[2*e] - 1
        let w = E[2*e + 1] - 1
        let dx = CX[u] - CX[w]
        let dy = CY[u] - CY[w]
        let d2 = dx*dx + dy*dy
        var err = d2 - {TARGET}
        if err < 0 {{ err = 0 - err }}
        if err < {TOL} {{ ok = ok + 1 }}
        e = e + 1
    }}
    return ok
}}

// 4-coloring CNF: var(v,c) = (v-1)*4 + c + 1, v in 1..{NV}, c in 0..3.
fn vc(v: i32, c: i32) -> i64 {{ return ((v - 1) * 4 + c + 1) as i64 }}

fn build_coloring() with Mut, Panic {{
    var v: i32 = 1
    while v <= {NV} {{
        LEARNT[0] = vc(v, 0)  LEARNT[1] = vc(v, 1)  LEARNT[2] = vc(v, 2)  LEARNT[3] = vc(v, 3)
        let _ = db_add(&LEARNT, 4)
        v = v + 1
    }}
    var e: i32 = 0
    while e < {NE} {{
        let u = E[2*e]
        let w = E[2*e + 1]
        var c: i32 = 0
        while c < 4 {{
            LEARNT[0] = 0 - vc(u, c)
            LEARNT[1] = 0 - vc(w, c)
            let _ = db_add(&LEARNT, 2)
            c = c + 1
        }}
        e = e + 1
    }}
}}

fn main() -> i32 with IO, Mut, Div, Panic {{
    print("=== de Grey unit-distance graph -> chi(R^2) >= 5 attempt ===\\n")
    print("vertices={NV}  edges={NE}  (data: de Grey 2018)\\n")
    ginit_all()
    let okg = validate_geometry()
    print("[geometry] unit-distance edges (|d2-1|<tol, exact i64, scale 1e7): ")
    print_dec(okg as i64)  print(" / {NE}\\n")
    if okg == {NE} {{ print("[geometry] PASS: all edges are unit-distance.\\n") }}
    else {{ print("[geometry] FAIL.\\n") }}

    build_coloring()
    let orig_nc = nclauses
    print("[cnf] 4-coloring CNF: vars="); print_dec(nvars as i64)
    print(" clauses="); print_dec(orig_nc as i64); print("\\n")
    print("[solve] CDCL (1-UIP, naive full-scan prop, no watches/VSIDS) budget=")
    print_dec(CONFLICT_CAP); print(" conflicts...\\n")

    let r = solve_cdcl()
    print("[solve] result code="); print_dec(r as i64)
    print("  conflicts="); print_dec(n_conflicts)
    print("  decisions="); print_dec(n_decisions)
    print("  lemmas="); print_dec(P_n as i64); print("\\n")

    if r == 0 {{
        print("[solve] UNSAT => this unit-distance graph is NOT 4-colorable => chi(R^2) >= 5\\n")
        reset_assign_full()
        let vr = verify(orig_nc)
        if vr == 1 {{ print("[verify] native RUP checker: VERIFIED\\n") }}
        else {{ print("[verify] native RUP: FAILED code="); print_dec(vr as i64); print("\\n") }}
    }} else if r == 1 {{
        print("[solve] SAT: a 4-coloring exists (this graph alone does not force chi>=5)\\n")
    }} else if r == 2 {{
        print("[solve] WALL: hit conflict budget (naive solver). No claim made.\\n")
    }} else {{
        print("[solve] capacity exhausted (arrays). No claim made.\\n")
    }}
    return 0
}}
"""

e_fns, e_names = emit_init("E", flatE, "init_E_")
cx_fns, cx_names = emit_init("CX", CX, "init_CX_")
cy_fns, cy_names = emit_init("CY", CY, "init_CY_")
all_init_names = e_names + cx_names + cy_names
ginit_calls = "\n".join(f"    {nm}()" for nm in all_init_names)
ginit = f"fn ginit_all() with Mut, Panic {{\n{ginit_calls}\n}}\n"

with open(OUT, "w") as o:
    o.write("// examples/erdos/degrey_chi5.sio  (GENERATED by examples/erdos/data/degrey/gen_sio.py)\n")
    o.write(f"// de Grey 2018 unit-distance graph: {NV} vertices, {NE} edges.\n")
    o.write("// Coordinates translated by (+5,+1) then scaled by 1e7 to positive i64 (no unary\n")
    o.write("// minus); translation+scaling preserves the unit-distance test exactly.\n")
    o.write(f"// Geometry target d2 = {TARGET} (= 1.0 * 1e7^2), tolerance {TOL}.\n")
    o.write("// NOTE: data arrays are filled at runtime (ginit_all) because this compiler's\n")
    o.write("// global-var literal-array initialiser mis-indexes (returns element 0).\n\n")
    o.write(f"var E: [i32; {2*NE}] = [0; {2*NE}]\n")
    o.write(f"var CX: [i64; {NV}] = [0; {NV}]\n")
    o.write(f"var CY: [i64; {NV}] = [0; {NV}]\n\n")
    o.write(e_fns)
    o.write(cx_fns)
    o.write(cy_fns)
    o.write(ginit)
    o.write(LOGIC)

print(f"wrote {OUT}: NV={NV} NE={NE} init_fns={len(all_init_names)} clauses_est={NV + NE*4}")

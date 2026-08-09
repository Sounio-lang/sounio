#!/usr/bin/env julia
# C5 — Julia validator (independent of the PyTorch pipeline).
#
# Validates, without importing any Python state:
#   V1. Golden corruption vectors: SEMANTIC validity against the C2 freeze
#       spec (Task A flip transitions; Task B reachability by swap or
#       mirror fallback; count invariance) + the file's internal
#       vectors_sha256 (Python json.dumps canonical form reimplemented).
#   V2. Receipt chain: sha256 of freeze artifacts vs recorded hashes;
#       sha256 sidecar of every results/*.json.
#   V3. Promotion gates: independent recomputation of the six frozen gates
#       from raw result JSONs (paired 95% CIs over 20 seeds, primary L=128).
#
# Usage: julia validator.jl <confirmatory_dir>
using SHA
using JSON

const CONF = length(ARGS) >= 1 ? ARGS[1] : "."
const OPEN, CLOSE, DOT = 1, 2, 0
const T95_19 = 2.093
const MODELS = ["CountBaseline", "RealTree-8", "CliffTree-8",
                "LearnedBilinTree", "OctTree-8", "GRU-8"]
const LS = [64, 128, 256, 512]
const FAILS = String[]
const REPORT = Dict{String,Any}("gates" => Dict{String,Any}(),
                                "cells_verified" => 0)

fail(msg) = (push!(FAILS, msg); println("  FAIL: $msg"))
ok(msg) = println("  ok: $msg")

decode(s) = [c == '(' ? OPEN : c == ')' ? CLOSE : DOT for c in s]
encode(t) = join(c == OPEN ? "(" : c == CLOSE ? ")" : "." for c in t)

function matched_pairs(tokens)
    stack = Int[]; pairs = Dict{Int,Int}(); segments = Tuple{Int,Int}[]
    seg_start = 0
    for (i, t) in enumerate(tokens)
        if t == OPEN
            isempty(stack) && (seg_start = i)
            push!(stack, i)
        elseif t == CLOSE && !isempty(stack)
            j = pop!(stack); pairs[j] = i
            isempty(stack) && push!(segments, (seg_start, i))
        end
    end
    pairs, segments
end

function swap_outputs(tokens)
    """All strings reachable by one Task B swap of qualifying segments."""
    _, segments = matched_pairs(tokens)
    outs = Set{String}()
    for a in 1:length(segments), b in a+1:length(segments)
        (s1, e1), (s2, e2) = segments[a], segments[b]
        la, lb = e1 - s1 + 1, e2 - s2 + 1
        0.5 <= la / lb <= 2.0 || continue
        out = vcat(tokens[1:s1-1], tokens[s2:e2], tokens[e1+1:s2-1],
                   tokens[s1:e1], tokens[e2+1:end])
        push!(outs, encode(out))
    end
    outs
end

function mirror_output(tokens)
    pairs, _ = matched_pairs(tokens)
    isempty(pairs) && return encode(tokens)
    j = sort(collect(keys(pairs)),
             by = j -> (-(pairs[j] - j), j))[1]
    i = pairs[j]
    seg = reverse(tokens[j:i])
    seg = [t == OPEN ? CLOSE : t == CLOSE ? OPEN : t for t in seg]
    encode(vcat(tokens[1:j-1], seg, tokens[i+1:end]))
end

# ---- V1: golden vectors ---------------------------------------------------
println("== V1: golden corruption vectors ==")
golden = JSON.parsefile(joinpath(CONF, "golden_corruptions.json"))
for v in golden["vectors"]
    inp, a_out, b_out = v["input"], v["task_a_output"], v["task_b_output"]
    tin = decode(inp)
    # Task A: length preserved, Hamming <= n_flip (positions may be re-hit)
    nf = max(1, length(tin) ÷ 8)
    ta = decode(a_out)
    length(ta) == length(tin) || fail("A length changed: $inp")
    sum(ta .!= tin) <= nf || fail("A mutated more than n_flip: $inp")
    # Task B: count invariance + reachability
    tb = decode(b_out)
    for s in (OPEN, CLOSE, DOT)
        count(==(s), tb) == count(==(s), tin) ||
            fail("B count invariance violated ($s): $inp")
    end
    reachable = swap_outputs(tin)
    push!(reachable, mirror_output(tin))
    b_out in reachable || fail("B output not reachable by spec: $inp -> $b_out")
end
ok("$(length(golden["vectors"])) vectors semantically valid")

# vectors_sha256: recompute Python json.dumps(vectors, indent=2, sort_keys=True)
function py_json(v, lvl)
    sp(l) = "  " ^ l
    if v isa AbstractDict
        ks = sort(collect(keys(v)))
        parts = [sp(lvl + 1) * "\"$k\": " * py_json(v[k], lvl + 1) for k in ks]
        return "{\n" * join(parts, ",\n") * "\n" * sp(lvl) * "}"
    elseif v isa AbstractVector
        isempty(v) && return "[]"
        parts = [sp(lvl + 1) * py_json(x, lvl + 1) for x in v]
        return "[\n" * join(parts, ",\n") * "\n" * sp(lvl) * "]"
    elseif v isa AbstractString
        return JSON.json(v)
    elseif v isa Bool
        return v ? "true" : "false"
    elseif v isa Integer
        return string(v)
    elseif v isa AbstractFloat
        return JSON.json(v)
    else
        error("unsupported $(typeof(v))")
    end
end
raw = py_json(golden["vectors"], 0)
digest = bytes2hex(sha256(raw))
digest == golden["vectors_sha256"] ||
    fail("vectors_sha256 mismatch: $digest != $(golden["vectors_sha256"])")
ok("vectors_sha256 matches ($(digest[1:12])…)")

# ---- V2: receipt chain ----------------------------------------------------
println("== V2: receipt chain ==")
const KNOWN = Dict(
    "freeze/manifest.json" => "50668b60646b02378475e343a74b76d9c5f4a0e2de51433f6ff68c8d600acb18",
)
for (rel, want) in KNOWN
    p = joinpath(CONF, rel)
    got = bytes2hex(open(sha256, p))
    got == want || fail("$rel sha256 $got != recorded $want")
    ok("$rel sha256 verified")
end
# SHA256SUMS internal consistency
sumsfile = joinpath(CONF, "freeze", "SHA256SUMS")
if isfile(sumsfile)
    for line in eachline(sumsfile)
        isempty(strip(line)) && continue
        h, f = split(strip(line), r"\s+"; limit = 2)
        p = joinpath(CONF, "freeze", f)
        isfile(p) || (fail("SHA256SUMS entry missing: $f"); continue)
        got = bytes2hex(open(sha256, p))
        got == h || fail("SHA256SUMS mismatch: $f")
    end
    ok("freeze/SHA256SUMS internally consistent")
end
# result sidecars
resdir = joinpath(CONF, "results")
cells = Dict{Tuple{Int,Int},Any}()
if isdir(resdir)
    for f in sort(readdir(resdir))
        m = match(r"^seed(\d+)_L(\d+)\.json$", f)
        m === nothing && continue
        p = joinpath(resdir, f)
        side = p[1:end-5] * ".sha256"
        isfile(side) || (fail("missing sidecar: $f"); continue)
        got = bytes2hex(open(sha256, p))
        recorded = split(read(side, String))[1]
        got == recorded || (fail("sidecar mismatch: $f"); continue)
        d = JSON.parsefile(p)
        cells[(d["seed_idx"], d["L"])] = d
    end
    ok("$(length(cells)) result cells hash-verified")
    REPORT["cells_verified"] = length(cells)
end

# ---- V3: promotion gates --------------------------------------------------
println("== V3: promotion gates (independent recomputation) ==")
ci_mean(xs) = (m = sum(xs) / length(xs);
               length(xs) < 2 ? (m, NaN) :
               (m, T95_19 * sqrt(sum((x - m)^2 for x in xs) / (length(xs) - 1) / length(xs))))

series(L, arm, model) = [cells[(i, L)]["arms"][arm][model]["test_acc_final"]
                         for i in 0:19 if haskey(cells, (i, L))]
pdiff(L, arm, m1, m2) = [cells[(i, L)]["arms"][arm][m1]["test_acc_final"] -
                         cells[(i, L)]["arms"][arm][m2]["test_acc_final"]
                         for i in 0:19 if haskey(cells, (i, L))]

for L in LS
    isempty(series(L, "B", "OctTree-8")) && continue
    n = length(series(L, "B", "OctTree-8"))
    println("  -- L=$L (n=$n seeds)")
    m1, h1 = ci_mean(pdiff(L, "B", "OctTree-8", "CliffTree-8"))
    m2, h2 = ci_mean(pdiff(L, "B", "OctTree-8", "RealTree-8"))
    m3, _ = ci_mean(pdiff(L, "B", "OctTree-8", "LearnedBilinTree"))
    m4, _ = ci_mean(pdiff(L, "A", "OctTree-8", "CliffTree-8"))
    m4r, _ = ci_mean(pdiff(L, "A", "OctTree-8", "RealTree-8"))
    mcb, _ = ci_mean(series(L, "B", "CountBaseline"))
    chance_half = 1.96 * sqrt(0.25 / 4096)
    negok = all(abs(ci_mean(series(L, "NEG", m))[1] - 0.5) <= chance_half
                for m in MODELS if !isempty(series(L, "NEG", m)))
    g1 = m1 - h1 > 0; g2 = m2 - h2 > 0; g3 = m3 >= -0.02
    g4 = m4 > 0 && m4r > 0; g5 = mcb <= 0.55
    println("   G1 Oct-Cliff $(round(m1, digits=4))±$(round(h1, digits=4)) pass=$g1")
    println("   G2 Oct-Real  $(round(m2, digits=4))±$(round(h2, digits=4)) pass=$g2")
    println("   G3 Oct-Learn $(round(m3, digits=4)) pass=$g3 (>= -2pp)")
    println("   G4 A-dir     cliff $(round(m4, digits=4)) real $(round(m4r, digits=4)) pass=$g4")
    println("   G5 CountBase $(round(mcb, digits=4)) pass=$g5")
    println("   G6 NEG       pass=$negok")
    REPORT["gates"][string(L)] = Dict(
        "n_seeds" => n,
        "G1" => Dict("diff" => m1, "ci" => h1, "pass" => g1),
        "G2" => Dict("diff" => m2, "ci" => h2, "pass" => g2),
        "G3" => Dict("diff" => m3, "pass" => g3),
        "G4" => Dict("cliff" => m4, "real" => m4r, "pass" => g4),
        "G5" => Dict("countbaseline" => mcb, "pass" => g5),
        "G6" => Dict("pass" => negok),
        "all_pass" => g1 && g2 && g3 && g4 && g5 && g6,
    )
    n == 20 && L == 128 &&
        println("   PROMOTION (primary): $(g1 && g2 && g3 && g4 && g5 && g6)")
end

println()
REPORT["fails"] = FAILS
REPORT["verdict"] = isempty(FAILS) ? "PASS" : "FAIL"
if haskey(REPORT["gates"], "128") && REPORT["gates"]["128"]["n_seeds"] == 20
    g = REPORT["gates"]["128"]
    sign_ok = all(haskey(REPORT["gates"], string(L)) &&
                  REPORT["gates"][string(L)]["G1"]["diff"] > 0 &&
                  REPORT["gates"][string(L)]["G2"]["diff"] > 0 for L in LS)
    REPORT["promotion"] = Dict("primary_L128_gates_pass" => g["all_pass"],
                               "sign_consistency_all_L" => sign_ok,
                               "claim_promoted" => g["all_pass"] && sign_ok)
end
resdir = joinpath(CONF, "results")
isdir(resdir) || mkpath(resdir)
open(joinpath(resdir, "validator_c5_report.json"), "w") do io
    JSON.print(io, REPORT, 2)
    write(io, "\n")
end
println("wrote $(joinpath(resdir, "validator_c5_report.json"))")
if isempty(FAILS)
    println("VALIDATOR: all checks passed ($(length(cells)) cells)")
else
    println("VALIDATOR: $(length(FAILS)) failure(s)")
    exit(1)
end

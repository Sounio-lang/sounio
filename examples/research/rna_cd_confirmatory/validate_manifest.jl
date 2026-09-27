#!/usr/bin/env julia

# Independent Base/stdlib-only validator for the Sounio RNA-CD manifest.

using SHA

const SCHEMA_VERSION = "rna-cd-manifest-v0.2.0"
const HASH_BASE = 131
const HASH_MODULUS = 2_147_483_647
const OUTER_FOLDS = 5
const INPUT_HEADER = [
    "record_id", "family", "clan_or_NO_CLAN", "sequence", "structure",
    "sequence_sha256", "structure_sha256",
]
const OUTPUT_HEADER = [
    "schema_version", "record_id", "family", "group_id", "length",
    "gc_count", "pair_count", "structural_stratum",
    "crossing_relation_count", "crossing_pair_count",
    "crossing_component_count", "maximum_crossing_degree", "outer_fold",
    "sequence_sha256", "structure_sha256",
]
const RECORD_RE = r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
const FAMILY_RE = r"^RF[0-9]+$"
const CLAN_RE = r"^CL[0-9]+$"
const SHA256_RE = r"^[0-9a-f]{64}$"
const RNA_ALPHABET = Set(codeunits("ACGUN"))
const STRUCTURE_ALPHABET = Set(codeunits(".()[]{}<>"))

struct Record
    record_id::String
    family::String
    group_id::String
    sequence_sha256::String
    structure_sha256::String
    length::Int
    gc_count::Int
    pair_count::Int
    structural_stratum::String
    crossing_relation_count::Int
    crossing_pair_count::Int
    crossing_component_count::Int
    maximum_crossing_degree::Int
    outer_fold::Int
end

refuse(message::AbstractString) = error("RNA_CD_JULIA_REFUSE " * message)

function strict_lines(path::String, label::String)
    isfile(path) || refuse("$label missing path=$path")
    bytes = read(path)
    isempty(bytes) && refuse("$label empty")
    bytes[end] == UInt8('\n') || refuse("$label missing_terminal_lf")
    any(==(UInt8('\r')), bytes) && refuse("$label contains_cr")
    text = try
        String(bytes)
    catch
        refuse("$label invalid_utf8")
    end
    lines = String.(split(text, '\n'; keepempty=true))
    pop!(lines)
    any(isempty, lines) && refuse("$label blank_line")
    lines
end

function fields(line::AbstractString, count::Int, label::String, line_number::Int)
    values = String.(split(line, '\t'; keepempty=true))
    length(values) == count ||
        refuse("$label line=$line_number field_count=$(length(values)) expected=$count")
    any(isempty, values) && refuse("$label line=$line_number empty_field")
    values
end

sha256hex(value::String) = bytes2hex(sha256(codeunits(value)))

function split_hash(value::String)
    h = Int64(0)
    for byte in codeunits(value)
        h = mod(h * HASH_BASE + Int64(byte), HASH_MODULUS)
    end
    h
end

function crossing_root!(parents::Vector{Int}, index::Int)
    root = index
    while parents[root] != root
        root = parents[root]
    end
    while parents[index] != index
        next_index = parents[index]
        parents[index] = root
        index = next_index
    end
    root
end

function crossing_union!(parents::Vector{Int}, first::Int, second::Int)
    first_root = crossing_root!(parents, first)
    second_root = crossing_root!(parents, second)
    first_root == second_root || (parents[second_root] = first_root)
end

function structure_metrics(structure::String, line_number::Int)
    all(byte -> byte in STRUCTURE_ALPHABET, codeunits(structure)) ||
        refuse("input line=$line_number invalid_structure_symbol")
    stacks = Dict{UInt8,Vector{Int}}(
        UInt8('(') => Int[], UInt8('[') => Int[],
        UInt8('{') => Int[], UInt8('<') => Int[],
    )
    opener = Dict{UInt8,UInt8}(
        UInt8(')') => UInt8('('), UInt8(']') => UInt8('['),
        UInt8('}') => UInt8('{'), UInt8('>') => UInt8('<'),
    )
    pairs = Tuple{Int,Int}[]
    for (position, byte) in enumerate(codeunits(structure))
        if haskey(stacks, byte)
            push!(stacks[byte], position)
        elseif haskey(opener, byte)
            key = opener[byte]
            isempty(stacks[key]) && refuse("input line=$line_number unmatched_close")
            push!(pairs, (pop!(stacks[key]), position))
        elseif byte != UInt8('.')
            refuse("input line=$line_number invalid_structure_symbol")
        end
    end
    all(isempty, values(stacks)) || refuse("input line=$line_number unclosed_pair")

    pair_count = length(pairs)
    degrees = zeros(Int, pair_count)
    parents = collect(1:pair_count)
    crossing_relation_count = 0
    for first in 1:pair_count
        a, b = pairs[first]
        for second in (first + 1):pair_count
            c, d = pairs[second]
            crosses = (a < c < b < d) || (c < a < d < b)
            if crosses
                crossing_relation_count += 1
                degrees[first] += 1
                degrees[second] += 1
                crossing_union!(parents, first, second)
            end
        end
    end
    crossing_pair_count = count(>(0), degrees)
    maximum_crossing_degree = isempty(degrees) ? 0 : maximum(degrees)
    roots = Set{Int}()
    for index in eachindex(degrees)
        degrees[index] > 0 && push!(roots, crossing_root!(parents, index))
    end
    crossing_component_count = length(roots)
    structural_stratum = crossing_relation_count > 0 ? "crossing" : "nested"
    (
        pair_count,
        structural_stratum,
        crossing_relation_count,
        crossing_pair_count,
        crossing_component_count,
        maximum_crossing_degree,
    )
end

function parse_input(path::String, salt::String)
    occursin(RECORD_RE, salt) || refuse("invalid_salt")
    lines = strict_lines(path, "input")
    fields(lines[1], length(INPUT_HEADER), "input", 1) == INPUT_HEADER ||
        refuse("input header")
    length(lines) > 1 || refuse("input no_records")

    records = Record[]
    previous_id = ""
    family_group = Dict{String,String}()
    sequence_group = Dict{String,String}()

    for (offset, line) in enumerate(lines[2:end])
        line_number = offset + 1
        record_id, family, clan, sequence, structure, sequence_sha, structure_sha =
            fields(line, 7, "input", line_number)
        occursin(RECORD_RE, record_id) || refuse("input line=$line_number record_id")
        occursin(FAMILY_RE, family) || refuse("input line=$line_number family")
        (clan == "NO_CLAN" || occursin(CLAN_RE, clan)) ||
            refuse("input line=$line_number clan")
        all(byte -> byte in RNA_ALPHABET, codeunits(sequence)) ||
            refuse("input line=$line_number sequence")
        ncodeunits(sequence) == ncodeunits(structure) ||
            refuse("input line=$line_number length_mismatch")
        occursin(SHA256_RE, sequence_sha) || refuse("input line=$line_number sequence_sha_format")
        occursin(SHA256_RE, structure_sha) || refuse("input line=$line_number structure_sha_format")
        sha256hex(sequence) == sequence_sha || refuse("input line=$line_number sequence_sha_mismatch")
        sha256hex(structure) == structure_sha || refuse("input line=$line_number structure_sha_mismatch")
        isempty(previous_id) || previous_id < record_id ||
            refuse("input line=$line_number record_order")
        previous_id = record_id

        group_id = clan == "NO_CLAN" ? "family:" * family : "clan:" * clan
        previous_group = get(family_group, family, group_id)
        previous_group == group_id || refuse("input line=$line_number family_group_conflict")
        family_group[family] = group_id
        duplicate_group = get(sequence_group, sequence_sha, group_id)
        duplicate_group == group_id || refuse("input line=$line_number duplicate_cross_group")
        sequence_group[sequence_sha] = group_id

        pair_count, structural_stratum, crossing_relation_count,
            crossing_pair_count, crossing_component_count,
            maximum_crossing_degree = structure_metrics(structure, line_number)
        pair_count >= 1 || refuse("input line=$line_number no_maskable_pair")
        fold = Int(mod(split_hash(salt * "|" * group_id), OUTER_FOLDS))
        push!(records, Record(
            record_id,
            family,
            group_id,
            sequence_sha,
            structure_sha,
            ncodeunits(sequence),
            count(byte -> byte == UInt8('C') || byte == UInt8('G'), codeunits(sequence)),
            pair_count,
            structural_stratum,
            crossing_relation_count,
            crossing_pair_count,
            crossing_component_count,
            maximum_crossing_degree,
            fold,
        ))
    end
    records
end

function render(records)
    lines = String[join(OUTPUT_HEADER, '\t')]
    for record in records
        push!(lines, join((
            SCHEMA_VERSION,
            record.record_id,
            record.family,
            record.group_id,
            string(record.length),
            string(record.gc_count),
            string(record.pair_count),
            record.structural_stratum,
            string(record.crossing_relation_count),
            string(record.crossing_pair_count),
            string(record.crossing_component_count),
            string(record.maximum_crossing_degree),
            string(record.outer_fold),
            record.sequence_sha256,
            record.structure_sha256,
        ), '\t'))
    end
    Vector{UInt8}(codeunits(join(lines, '\n') * "\n"))
end

function validate(input_path::String, artifact_path::String, salt::String)
    records = parse_input(input_path, salt)
    artifact_lines = strict_lines(artifact_path, "artifact")
    fields(artifact_lines[1], length(OUTPUT_HEADER), "artifact", 1) == OUTPUT_HEADER ||
        refuse("artifact header")
    length(artifact_lines) == length(records) + 1 || refuse("artifact row_count")
    expected = render(records)
    actual = read(artifact_path)
    actual == expected || refuse("artifact byte_mismatch")
    groups = length(unique(record.group_id for record in records))
    nested = count(record -> record.structural_stratum == "nested", records)
    crossing = count(record -> record.structural_stratum == "crossing", records)
    println("RNA_CD_JULIA_VALIDATION_PASS records=$(length(records)) groups=$groups nested=$nested crossing=$crossing tolerance=0")
end

function must_reject(f, label)
    try
        f()
    catch
        return
    end
    error("self-test failed to reject $label")
end

function self_test()
    split_hash("rna-cd-confirmatory-v1|clan:CL00001") == 670_237_609 ||
        error("split hash drift")
    structure_metrics("((..))", 1) == (2, "nested", 0, 0, 0, 0) ||
        error("nested metrics")
    structure_metrics("([..)]", 1) == (2, "crossing", 1, 2, 1, 1) ||
        error("crossing metrics")
    structure_metrics("([<{)]>}.", 1) == (4, "crossing", 6, 4, 1, 3) ||
        error("multi-crossing metrics")
    structure_metrics("([)]..{<}>", 1) == (4, "crossing", 2, 4, 2, 1) ||
        error("two-component crossing metrics")
    structure_metrics("(([.)])", 1) == (3, "crossing", 1, 2, 1, 1) ||
        error("mixed nested-crossing metrics")
    must_reject(() -> structure_metrics("(()...", 1), "unclosed structure")
    must_reject(() -> structure_metrics(").....", 1), "unmatched close")
    println("RNA_CD_JULIA_SELF_TEST_PASS")
end

function main()
    if ARGS == ["--self-test"]
        self_test()
        return
    end
    length(ARGS) == 3 || begin
        println(stderr, "usage: validate_manifest.jl <records.tsv> <manifest.tsv> <salt>")
        exit(2)
    end
    try
        validate(ARGS[1], ARGS[2], ARGS[3])
    catch exception
        println(stderr, exception)
        exit(1)
    end
end

main()

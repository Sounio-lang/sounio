# Extraction for checker_spine_parity_gate.sh. Pure awk: this tree's rule is
# that Python is a binding or a marshaller, never computation of our own, and
# a CI gate's analysis is computation of our own.
#
# Emits, on stdout:
#   BYVAL <ItemKind>      for each arm of the by-value collect_item
#   MUT   <ItemKind>      for each arm of the *mut checker_collect_item_inplace
#   RETCK <name>          for each impl-Checker method returning a Checker
#   BRIDGE <fn>-><method> for each (*c).method( inside a *mut function
#
# Line comments are stripped first. An earlier version did not strip them and
# reported a bridge that existed only inside a comment documenting that
# bridge's removal -- the gate failing on its own prose.

function decomment(line,   i, before) {
    i = index(line, "//")
    if (i == 0) return line
    before = substr(line, 1, i - 1)
    # crude string guard: an odd number of quotes before // means we are inside one
    if (gsub(/"/, "\"", before) % 2 == 1) return line
    return before
}

function braces(line,   n, o, c) {
    o = gsub(/\{/, "{", line)
    c = gsub(/\}/, "}", line)
    return o - c
}

{
    line = decomment($0)

    # --- enter / leave the two collect bodies ---
    if (line ~ /^[[:space:]]*fn collect_item\(self, item: Item\)/) { inby = 1; bydepth = 0 }
    if (line ~ /^fn checker_collect_item_inplace\(c: \*mut Checker/) { inmut = 1; mutdepth = 0 }

    if (inby) {
        if (match(line, /ItemKind::Item[A-Za-z]+/))
            print "BYVAL " substr(line, RSTART + 10, RLENGTH - 10)
        bydepth += braces(line)
        if (bydepth <= 0 && line ~ /\{/) { }
        if (bydepth <= 0 && seenby) inby = 0
        if (line ~ /\{/) seenby = 1
    }
    if (inmut) {
        if (match(line, /ItemKind::Item[A-Za-z]+/))
            print "MUT " substr(line, RSTART + 10, RLENGTH - 10)
        mutdepth += braces(line)
        if (mutdepth <= 0 && seenmut) inmut = 0
        if (line ~ /\{/) seenmut = 1
    }

    # --- by-value methods that return a Checker ---
    if (match(line, /^[[:space:]]*fn [a-z_0-9]+\(self/)) {
        name = line
        sub(/^[[:space:]]*fn /, "", name)
        sub(/\(.*$/, "", name)
        if (line ~ /->[[:space:]]*\(?Checker/) print "RETCK " name
    }

    # --- track the enclosing *mut function, then look for bridges ---
    if (match(line, /^[[:space:]]*(pub )?fn [a-z_0-9]+\(/)) {
        fname = line
        sub(/^[[:space:]]*(pub )?fn /, "", fname)
        sub(/\(.*$/, "", fname)
        cur = (line ~ /c: \*mut Checker/) ? fname : ""
    }
    if (cur != "") {
        rest = line
        while (match(rest, /\(\*c\)\.[a-z_0-9]+\(/)) {
            call = substr(rest, RSTART + 5, RLENGTH - 6)
            print "BRIDGE " cur "->" call
            rest = substr(rest, RSTART + RLENGTH)
        }
    }
}

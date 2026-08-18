#!/usr/bin/env bash
# agent-bus — a shared, append-only channel between the agents on this pod.
#
# WHY THIS EXISTS. Coordination here was a document (.claude/AGENT_HANDOFF.md),
# last written by a human five weeks before this file was created. Documents do
# not tell you that another agent is holding the build lock right now, or that
# SOUC_BIN is poisoned in the pod environment and every gate you run is silently
# measuring somebody else's checkout. Both of those cost real hours.
#
# WHAT IT IS AND IS NOT. It is a shared append-only log plus expiring leases on
# a filesystem every agent already mounts. It is NOT push: nothing interrupts
# another agent's loop. An agent hears you when it reads, so the contract is
#
#     read `brief` before you start anything, post when state changes.
#
# That is the whole protocol. It is cheap enough to obey and useless if not.
#
# Storage lives OUTSIDE any git checkout, at /workspace/.agents/bus, because
# agents work in different worktrees on different branches and a channel that
# lives in one of them is invisible to the others.
set -uo pipefail

BUS="${AGENT_BUS_DIR:-/workspace/.agents/bus}"
EVENTS="$BUS/events.jsonl"
LEASES="$BUS/leases"
HAZARDS="$BUS/hazards"

# Identity comes from the agent's own HOME, which the pod already assigns per
# slot (/workspace/.home/.../.agents/claude-1). Override with AGENT_ID.
ME="${AGENT_ID:-$(basename "${HOME:-unknown}")}"

mkdir -p "$BUS" "$LEASES" "$HAZARDS" 2>/dev/null
[[ -f "$EVENTS" ]] || : > "$EVENTS"
chmod -R a+rwX "$BUS" 2>/dev/null || true

now()      { date -u +%Y-%m-%dT%H:%M:%SZ; }
epoch()    { date -u +%s; }
slugify()  { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }
# JSON-escape: the message is free text and WILL contain quotes and backslashes.
jesc()     { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/ /g'; }

append_event() { # kind, text
    local kind="$1"; shift
    local text="$*"
    # Single printf under O_APPEND: atomic for a line this size, so concurrent
    # agents cannot interleave halves of each other's events.
    printf '{"ts":"%s","agent":"%s","kind":"%s","text":"%s"}\n' \
        "$(now)" "$(jesc "$ME")" "$(jesc "$kind")" "$(jesc "$text")" >> "$EVENTS"
}

fmt_events() { # reads jsonl on stdin
    sed -E 's/^\{"ts":"([^"]*)","agent":"([^"]*)","kind":"([^"]*)","text":"(.*)"\}$/\1|\2|\3|\4/' \
    | awk -F'|' '{ printf "%s  %-10s %-8s %s\n", substr($1,6,14), $2, $3, substr($0, index($0,$4)) }' \
    | sed -e 's/\\"/"/g' -e 's/\\\\/\\/g'
}

usage() {
    cat <<'EOF'
agent-bus — shared channel between the agents on this pod

  brief                      START HERE. Who is active, what is claimed,
                             what hazards are live, and the last 10 events.
  post <kind> <text...>      kind: status | finding | blocker | done
  tail [n]                   last n events (default 30)
  since <minutes>            events from the last N minutes
  who                        agents that posted in the last 2h, and their leases

  claim <resource> [ttl_min] take an exclusive lease (default 90 min).
                             Exits 1 if somebody else holds it — check first.
  release <resource>         drop your lease
  leases                     everything currently held, with time left

  hazard add <slug> <text>   an environment fact that will silently ruin other
                             agents' measurements (poisoned env var, stale ELF,
                             a checkout parked on another branch)
  hazard list
  hazard clear <slug>

Identity is $HOME's basename; override with AGENT_ID. Storage: /workspace/.agents/bus
EOF
}

cmd="${1:-brief}"; shift 2>/dev/null || true

case "$cmd" in
  post)
    kind="${1:-status}"; shift 2>/dev/null || true
    [[ $# -gt 0 ]] || { echo "agent-bus: post needs a message" >&2; exit 2; }
    append_event "$kind" "$*"
    echo "posted as $ME"
    ;;

  tail)
    n="${1:-30}"
    tail -n "$n" "$EVENTS" | fmt_events
    ;;

  since)
    mins="${1:-60}"
    cutoff=$(( $(epoch) - mins * 60 ))
    while IFS= read -r line; do
        ts=$(sed -E 's/^\{"ts":"([^"]*)".*/\1/' <<<"$line")
        e=$(date -u -d "$ts" +%s 2>/dev/null) || continue
        [[ $e -ge $cutoff ]] && printf '%s\n' "$line"
    done < "$EVENTS" | fmt_events
    ;;

  who)
    echo "== agents posting in the last 2h =="
    cutoff=$(( $(epoch) - 7200 ))
    while IFS= read -r line; do
        ts=$(sed -E 's/^\{"ts":"([^"]*)".*/\1/' <<<"$line")
        e=$(date -u -d "$ts" +%s 2>/dev/null) || continue
        [[ $e -ge $cutoff ]] && sed -E 's/.*"agent":"([^"]*)".*/\1/' <<<"$line"
    done < "$EVENTS" | sort | uniq -c | sort -rn | sed 's/^/  /'
    echo "== leases =="
    "$0" leases
    ;;

  claim)
    res="${1:?agent-bus: claim needs a resource}"; ttl="${2:-90}"
    d="$LEASES/$(slugify "$res")"
    # mkdir is atomic on this filesystem, so it IS the lock. Expiry is by mtime
    # so an agent that dies mid-task does not park a resource forever.
    if ! mkdir "$d" 2>/dev/null; then
        holder=$(sed -n '1p' "$d/meta" 2>/dev/null || echo unknown)
        exp=$(sed -n '2p' "$d/meta" 2>/dev/null || echo 0)
        if [[ "$exp" =~ ^[0-9]+$ ]] && [[ $(epoch) -gt $exp ]]; then
            echo "agent-bus: lease on '$res' from $holder EXPIRED — taking it"
            rm -rf "$d"; mkdir "$d" 2>/dev/null || { echo "agent-bus: lost the race" >&2; exit 1; }
        else
            left=$(( (exp - $(epoch)) / 60 ))
            echo "agent-bus: '$res' is held by $holder for another ${left}m" >&2
            exit 1
        fi
    fi
    printf '%s\n%s\n%s\n' "$ME" "$(( $(epoch) + ttl * 60 ))" "$(now)" > "$d/meta"
    chmod -R a+rwX "$d" 2>/dev/null || true
    append_event status "claimed $res for ${ttl}m"
    echo "claimed '$res' as $ME for ${ttl}m"
    ;;

  release)
    res="${1:?agent-bus: release needs a resource}"
    d="$LEASES/$(slugify "$res")"
    [[ -d "$d" ]] || { echo "agent-bus: '$res' is not held"; exit 0; }
    holder=$(sed -n '1p' "$d/meta" 2>/dev/null || echo unknown)
    if [[ "$holder" != "$ME" && -z "${AGENT_BUS_FORCE:-}" ]]; then
        echo "agent-bus: '$res' is $holder's, not yours (AGENT_BUS_FORCE=1 to override)" >&2
        exit 1
    fi
    rm -rf "$d"
    append_event status "released $res"
    echo "released '$res'"
    ;;

  leases)
    found=0
    for d in "$LEASES"/*/; do
        [[ -d "$d" ]] || continue
        found=1
        holder=$(sed -n '1p' "$d/meta" 2>/dev/null || echo unknown)
        exp=$(sed -n '2p' "$d/meta" 2>/dev/null || echo 0)
        left=$(( (exp - $(epoch)) / 60 ))
        state="${left}m left"; [[ $left -lt 0 ]] && state="EXPIRED"
        printf '  %-34s %-10s %s\n' "$(basename "$d")" "$holder" "$state"
    done
    [[ $found -eq 1 ]] || echo "  (none)"
    ;;

  hazard)
    sub="${1:-list}"; shift 2>/dev/null || true
    case "$sub" in
      add)
        slug="${1:?agent-bus: hazard add needs a slug}"; shift
        [[ $# -gt 0 ]] || { echo "agent-bus: hazard add needs a description" >&2; exit 2; }
        printf '%s\n%s\n%s\n' "$ME" "$(now)" "$*" > "$HAZARDS/$(slugify "$slug")"
        chmod a+rw "$HAZARDS/$(slugify "$slug")" 2>/dev/null || true
        append_event blocker "HAZARD $slug: $*"
        echo "hazard '$slug' recorded"
        ;;
      clear)
        slug="${1:?agent-bus: hazard clear needs a slug}"
        rm -f "$HAZARDS/$(slugify "$slug")"
        append_event status "hazard cleared: $slug"
        echo "hazard '$slug' cleared"
        ;;
      list|*)
        found=0
        for f in "$HAZARDS"/*; do
            [[ -f "$f" ]] || continue
            found=1
            printf '  [%s] %s (%s, %s)\n' "$(basename "$f")" "$(sed -n '3p' "$f")" \
                   "$(sed -n '1p' "$f")" "$(sed -n '2p' "$f")"
        done
        [[ $found -eq 1 ]] || echo "  (none)"
        ;;
    esac
    ;;

  brief)
    echo "=============================================================="
    echo " agent-bus brief for $ME    $(now)"
    echo "=============================================================="
    echo
    echo "HAZARDS (read these before trusting any measurement):"
    "$0" hazard list
    echo
    echo "LEASES:"
    "$0" leases
    echo
    echo "LAST 10 EVENTS:"
    tail -n 10 "$EVENTS" | fmt_events | sed 's/^/  /'
    echo
    echo "post when your state changes:  scripts/dev/agent-bus.sh post status '...'"
    ;;

  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac

# Focused exact arithmetic and scope audit

The independent verifier printed:

```text
CLASSIFICATION=WITNESS_TRANSVERSALITY_UNRESOLVED
PRODUCTION_BOUNDARY_REPRODUCED=true
ORIGINAL_SYMBOLIC_VARIABLES=6
WITNESS_EVENT_VERIFY=PASS
```

The mutation runner printed `MUTATIONS_REJECTED=22/22`.

Exact receipt fields follow. Treat each pair as `[lower, upper]`.

Production derivative:

```text
[-33624906207145922263559247780051227755959087408140433116320689732843135525767/113078212145816597093331040047546785012958969400039613319782796882727665664,
 77088798665811792192215953071914291195254778315200041791817214650640896702707/226156424291633194186662080095093570025917938800079226639565593765455331328]
```

Terminal derivative:

```text
[-16810854937519898702686896151713659861021538102822548074459838177136179256757/56539106072908298546665520023773392506479484700019806659891398441363832832,
 77082426550155652760862003444194825904512492394146686596685468319230533929261/226156424291633194186662080095093570025917938800079226639565593765455331328]
```

Receipt counts:

```text
production_time_depth=10
production_step=1/262144
terminal_time_depth=18
terminal_step=1/67108864
projection_attempts=38
each attempt status=SECOND_EVENT_COVER_UNRESOLVED
each attempt detail="upward Newton cover left 256 unresolved leaves; first=UPWARD_PREFILTER_UNRESOLVED; split_nodes=255"
first_positive_endpoint=null
implementation_checks_passed=true
implementation_check_count=39
```

Recompute from the exact derivative endpoints:

1. `100*(1-terminal_width/production_width)`.
2. terminal midpoint and radius.
3. `terminal_radius/terminal_midpoint`.

The report gives, respectively:

```text
0.008843634958699553 percent
midpoint 21.75265821188538
radius 319.0841178894043
ratio 14.668741391572123
```

Audit the inference: time refinement did not materially reduce this derivative
obstruction; a fixed-midpoint enclosure needs radius reduction strictly beyond
the ratio for a positive lower endpoint. Confirm that this does not prove
absence of a true return, impossibility of spatial reconditioning, final
six-variable dependence, a covering relation, recurrence, or chaos.

Return concrete arithmetic and BLOCKER/MAJOR/MINOR findings. Do not request the
full files; all fields needed for these claims are embedded above.

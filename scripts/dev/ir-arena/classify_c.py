import subprocess, re

def strip(line):
    out,i,n=[],0,len(line)
    while i<n:
        c=line[i]
        if c in '"\'':
            q=c; out.append(" "); i+=1
            while i<n:
                if line[i]=="\\": out.append(" "); i+=2; continue
                if line[i]==q: break
                out.append(" "); i+=1
            out.append(" "); i+=1
        else: out.append(c); i+=1
    s="".join(out); j=s.find("//")
    return s if j<0 else s[:j]

files=[f for f in subprocess.run(["git","diff","--name-only","0ae1ebff20^","0ae1ebff20"],
      capture_output=True,text=True).stdout.splitlines() if f.endswith(".sio")]

buckets={}
for p in files:
    lines=open(p).read().split("\n")
    for n,raw in enumerate(lines):
        if "ir_arena_store" not in raw: continue
        c=strip(raw)
        if c.count("(")-c.count(")")==0: continue
        t=c.rstrip()
        if   t.endswith(",)"): k="C2a  opener ends ',)'"
        elif t.endswith("()"): k="C2b  opener ends '()'"
        elif t.endswith("("):  k="C1   opener ends '(' (missing closer at end)"
        elif t.endswith("{"):  k="OK   struct literal (already repaired)"
        else:                  k=f"???  opener ends {t[-3:]!r}"
        buckets.setdefault(k,[]).append(f"{p}:{n+1}")

tot=0
for k in sorted(buckets):
    v=buckets[k]; tot+=len(v)
    print(f"{len(v):4d}  {k}")
    for s in v: print(f"          {s}")
print(f"\nTOTAL {tot}")

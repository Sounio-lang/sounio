# M10: live oversubscription (session/slot decoupling + harden)

## Gap

5th concurrent connection RSTs: SV_MAX_SESS=4, accept closes
when no free fd slot, and session id == psl slot == bank lane
everywhere (sv_begin, blist, turn). M9 pq is never fed by
live traffic, so live preemption is impossible by
construction. serve_preempt covers the engine only.

## Current

llm_serve: 4 sessions, 4 psl slots, 1:1 by id. sv_begin
copies ids into bank lane si and inits psl[si]. blist
collects GEN sessions (bn<=4, no overflow). pq always
empty live. 4-way concurrent lockstep verified 4/4 MATCH.

## M10 changes (1+2)

1. Harden: 8 sessions (fd/state arrays 4->8, SV_MAX_SESS=8);
   pump slots stay 4. Non-pump signatures widened
   mechanically (dead path, SV_PUMP==1 const).
2. Decouple: sv_begin parses only, then pushes
   (off=-1, nid, pmax, ss=session) to pq (429+close when
   full). admit_all binds: fresh-live entries copy
   ids[ss] into the freed bank lane; restores copy spill
   as today. sess_of_slot[4] maps slot->session (-1 free);
   blist built by slot scan (bn<=4 structural). Turn
   translates slot->session for every session array;
   pdone/rslot stay slot-indexed. Unbind on finish/fail.
3. Reap: disconnect purges pq entries of the session
   (svb_qpurge, also frees staged victim regions) and frees
   bound slots (truncate + off=-1 + unbind).

## Verification

Rebuilt server, 5-way burst (max_tokens 8): preempt lines
in log, all 5 finish, burst texts byte-identical to solo
baselines. Full CPU suite green. Receipt in plan-m10.

Out of scope: priority classes, cross-node spill, non-pump
path behavior (signatures only).

--------------------------- MODULE SounioFleet ---------------------------
EXTENDS Naturals, Integers, TLC

CONSTANTS MaxEvents, MaxCapabilities

VARIABLES online,
          phase,
          desired,
          startCapsIssued,
          startCapsConsumed,
          starts,
          argvAttested,
          decision,
          logSeq,
          anchorSeq,
          anchorVerified,
          checkpoint,
          checkpointSeq,
          handoff,
          handoffSeq,
          handoffCapsIssued,
          handoffCapsConsumed

vars == << online, phase, desired, startCapsIssued, startCapsConsumed, starts,
           argvAttested, decision, logSeq, anchorSeq, anchorVerified,
           checkpoint, checkpointSeq, handoff, handoffSeq,
           handoffCapsIssued, handoffCapsConsumed >>

PhaseSet == {"Absent", "Active", "Drifted"}
DecisionSet == {"start", "noop", "blocked"}
CheckpointSet == {"None", "Draft", "Verified"}
HandoffSet == {"None", "Prepared", "Accepted"}

Init ==
    /\ online = TRUE
    /\ phase = "Absent"
    /\ desired = TRUE
    /\ startCapsIssued = 0
    /\ startCapsConsumed = 0
    /\ starts = 0
    /\ argvAttested = FALSE
    /\ decision = "start"
    /\ logSeq = 0
    /\ anchorSeq = 0
    /\ anchorVerified = FALSE
    /\ checkpoint = "None"
    /\ checkpointSeq = 0
    /\ handoff = "None"
    /\ handoffSeq = 0
    /\ handoffCapsIssued = 0
    /\ handoffCapsConsumed = 0

CanAppend == logSeq < MaxEvents

DisableDesired ==
    /\ CanAppend
    /\ desired
    /\ phase = "Absent"
    /\ desired' = FALSE
    /\ decision' = "noop"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, startCapsIssued, startCapsConsumed, starts,
                     argvAttested, anchorSeq, anchorVerified, checkpoint,
                     checkpointSeq, handoff, handoffSeq, handoffCapsIssued,
                     handoffCapsConsumed >>

EnableDesired ==
    /\ CanAppend
    /\ ~desired
    /\ desired' = TRUE
    /\ decision' = "start"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, startCapsIssued, startCapsConsumed, starts,
                     argvAttested, anchorSeq, anchorVerified, checkpoint,
                     checkpointSeq, handoff, handoffSeq, handoffCapsIssued,
                     handoffCapsConsumed >>

IssueStartCapability ==
    /\ CanAppend
    /\ desired
    /\ phase = "Absent"
    /\ startCapsIssued < MaxCapabilities
    /\ startCapsIssued' = startCapsIssued + 1
    /\ logSeq' = logSeq + 1
    /\ decision' = "start"
    /\ UNCHANGED << phase, desired, startCapsConsumed, starts,
                     argvAttested, anchorSeq, anchorVerified, checkpoint,
                     checkpointSeq, handoff, handoffSeq,
                     handoffCapsIssued, handoffCapsConsumed >>

StartWithLinearCapability ==
    /\ CanAppend
    /\ desired
    /\ phase = "Absent"
    /\ startCapsConsumed < startCapsIssued
    /\ startCapsConsumed' = startCapsConsumed + 1
    /\ starts' = starts + 1
    /\ phase' = "Active"
    /\ argvAttested' = TRUE
    /\ decision' = "noop"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << desired, startCapsIssued, anchorSeq, anchorVerified,
                     checkpoint, checkpointSeq, handoff, handoffSeq,
                     handoffCapsIssued, handoffCapsConsumed >>

RefuseStartWithoutCapability ==
    /\ CanAppend
    /\ desired
    /\ phase = "Absent"
    /\ startCapsConsumed = startCapsIssued
    /\ decision' = "blocked"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, anchorSeq, anchorVerified,
                     checkpoint, checkpointSeq, handoff, handoffSeq,
                     handoffCapsIssued, handoffCapsConsumed >>

ObserveStop ==
    /\ CanAppend
    /\ phase = "Active"
    /\ phase' = "Absent"
    /\ argvAttested' = FALSE
    /\ decision' = "start"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << desired, startCapsIssued, startCapsConsumed, starts,
                     anchorSeq, anchorVerified, checkpoint, checkpointSeq,
                     handoff, handoffSeq, handoffCapsIssued,
                     handoffCapsConsumed >>

ObserveArgvDrift ==
    /\ CanAppend
    /\ phase = "Active"
    /\ phase' = "Drifted"
    /\ argvAttested' = FALSE
    /\ decision' = "blocked"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << desired, startCapsIssued, startCapsConsumed, starts,
                     anchorSeq, anchorVerified, checkpoint, checkpointSeq,
                     handoff, handoffSeq, handoffCapsIssued,
                     handoffCapsConsumed >>

RestoreAttestation ==
    /\ CanAppend
    /\ phase = "Drifted"
    /\ phase' = "Active"
    /\ argvAttested' = TRUE
    /\ decision' = "noop"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << desired, startCapsIssued, startCapsConsumed, starts,
                     anchorSeq, anchorVerified, checkpoint, checkpointSeq,
                     handoff, handoffSeq, handoffCapsIssued,
                     handoffCapsConsumed >>

CreateCheckpoint ==
    /\ CanAppend
    /\ phase = "Active"
    /\ argvAttested
    /\ checkpoint = "None"
    /\ checkpoint' = "Draft"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, decision, anchorSeq,
                     anchorVerified, checkpointSeq, handoff, handoffSeq,
                     handoffCapsIssued, handoffCapsConsumed >>

VerifyCheckpoint ==
    /\ CanAppend
    /\ checkpoint = "Draft"
    /\ checkpoint' = "Verified"
    /\ checkpointSeq' = logSeq + 1
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, decision, anchorSeq,
                     anchorVerified, handoff, handoffSeq,
                     handoffCapsIssued, handoffCapsConsumed >>

PrepareHandoff ==
    /\ CanAppend
    /\ checkpoint = "Verified"
    /\ handoff = "None"
    /\ handoffCapsIssued < 1
    /\ handoff' = "Prepared"
    /\ handoffSeq' = logSeq + 1
    /\ handoffCapsIssued' = handoffCapsIssued + 1
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, decision, anchorSeq,
                     anchorVerified, checkpoint, checkpointSeq,
                     handoffCapsConsumed >>

AnchorVerifiedPrefix ==
    /\ anchorSeq < logSeq
    /\ anchorSeq' = logSeq
    /\ anchorVerified' = TRUE
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, decision, logSeq, checkpoint,
                     checkpointSeq, handoff, handoffSeq, handoffCapsIssued,
                     handoffCapsConsumed >>

ObserveAnchorRemoval ==
    /\ CanAppend
    /\ anchorVerified
    /\ anchorSeq > 0
    /\ handoff # "Accepted"
    /\ anchorSeq' = 0
    /\ anchorVerified' = FALSE
    /\ decision' = "blocked"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, checkpoint, checkpointSeq, handoff,
                     handoffSeq, handoffCapsIssued, handoffCapsConsumed >>

ObserveSignatureSubstitution ==
    /\ CanAppend
    /\ anchorVerified
    /\ anchorSeq > 0
    /\ handoff # "Accepted"
    /\ anchorVerified' = FALSE
    /\ decision' = "blocked"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, anchorSeq, checkpoint, checkpointSeq,
                     handoff, handoffSeq, handoffCapsIssued,
                     handoffCapsConsumed >>

AcceptAnchoredHandoff ==
    /\ CanAppend
    /\ handoff = "Prepared"
    /\ handoffCapsConsumed < handoffCapsIssued
    /\ anchorVerified
    /\ anchorSeq >= handoffSeq
    /\ handoff' = "Accepted"
    /\ handoffCapsConsumed' = handoffCapsConsumed + 1
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, decision, anchorSeq,
                     anchorVerified, checkpoint, checkpointSeq, handoffSeq,
                     handoffCapsIssued >>

RefuseUnanchoredHandoff ==
    /\ CanAppend
    /\ handoff = "Prepared"
    /\ \/ ~anchorVerified
       \/ anchorSeq < handoffSeq
    /\ decision' = "blocked"
    /\ logSeq' = logSeq + 1
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, anchorSeq, anchorVerified,
                     checkpoint, checkpointSeq, handoff, handoffSeq,
                     handoffCapsIssued, handoffCapsConsumed >>

PersistentStep ==
    \/ DisableDesired
    \/ EnableDesired
    \/ IssueStartCapability
    \/ StartWithLinearCapability
    \/ RefuseStartWithoutCapability
    \/ ObserveStop
    \/ ObserveArgvDrift
    \/ RestoreAttestation
    \/ CreateCheckpoint
    \/ VerifyCheckpoint
    \/ PrepareHandoff
    \/ AnchorVerifiedPrefix
    \/ ObserveAnchorRemoval
    \/ ObserveSignatureSubstitution
    \/ AcceptAnchoredHandoff
    \/ RefuseUnanchoredHandoff

Crash ==
    /\ online
    /\ online' = FALSE
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, decision, logSeq, anchorSeq,
                     anchorVerified, checkpoint, checkpointSeq, handoff,
                     handoffSeq, handoffCapsIssued, handoffCapsConsumed >>

Recover ==
    /\ ~online
    /\ online' = TRUE
    /\ UNCHANGED << phase, desired, startCapsIssued, startCapsConsumed,
                     starts, argvAttested, decision, logSeq, anchorSeq,
                     anchorVerified, checkpoint, checkpointSeq, handoff,
                     handoffSeq, handoffCapsIssued, handoffCapsConsumed >>

Next ==
    \/ /\ online
       /\ online' = TRUE
       /\ PersistentStep
    \/ Crash
    \/ Recover

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ online \in BOOLEAN
    /\ phase \in PhaseSet
    /\ desired \in BOOLEAN
    /\ startCapsIssued \in 0..MaxCapabilities
    /\ startCapsConsumed \in 0..MaxCapabilities
    /\ starts \in 0..MaxCapabilities
    /\ argvAttested \in BOOLEAN
    /\ decision \in DecisionSet
    /\ logSeq \in 0..MaxEvents
    /\ anchorSeq \in 0..MaxEvents
    /\ anchorVerified \in BOOLEAN
    /\ checkpoint \in CheckpointSet
    /\ checkpointSeq \in 0..MaxEvents
    /\ handoff \in HandoffSet
    /\ handoffSeq \in 0..MaxEvents
    /\ handoffCapsIssued \in 0..1
    /\ handoffCapsConsumed \in 0..1

LinearStartAuthority ==
    /\ startCapsConsumed <= startCapsIssued
    /\ starts = startCapsConsumed

ActiveProcessHasFullArgvAttestation ==
    argvAttested <=> phase = "Active"

DriftFailsClosed ==
    phase = "Drifted" => decision = "blocked"

SignedAnchorIsLogPrefix ==
    anchorVerified =>
        /\ anchorSeq > 0
        /\ anchorSeq <= logSeq

VerifiedCheckpointPrecedesHandoff ==
    handoff \in {"Prepared", "Accepted"} =>
        /\ checkpoint = "Verified"
        /\ checkpointSeq < handoffSeq

LinearHandoffAuthority ==
    handoffCapsConsumed <= handoffCapsIssued

AcceptedHandoffIsAnchored ==
    handoff = "Accepted" =>
        /\ handoffCapsConsumed = 1
        /\ anchorVerified
        /\ anchorSeq >= handoffSeq
        /\ checkpointSeq < handoffSeq

\* Concrete capability publication, action requests, and crash recovery map
\* to stuttering steps. Issue/commit/verify/accept events map to the named
\* persistent actions above; Crash and Recover change no persisted fact.

\* @sabotage id=start-without-capability invariant=LinearStartAuthority control=capability_required
\* @sabotage id=capability-reuse invariant=LinearStartAuthority control=capability_reuse
\* @sabotage id=argv-substitution invariant=ActiveProcessHasFullArgvAttestation control=argv_sabotage
\* @sabotage id=generation-substitution invariant=DriftFailsClosed control=generation_sabotage
\* @sabotage id=anchor-removal invariant=SignedAnchorIsLogPrefix control=anchor_removal
\* @sabotage id=signature-substitution invariant=AcceptedHandoffIsAnchored control=signature_sabotage
\* @sabotage id=wrong-checkpoint-state invariant=VerifiedCheckpointPrecedesHandoff control=wrong_checkpoint_state
\* @sabotage id=unanchored-handoff invariant=AcceptedHandoffIsAnchored control=unanchored_handoff

=============================================================================

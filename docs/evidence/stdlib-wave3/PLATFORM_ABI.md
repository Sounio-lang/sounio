# PollSet, UDP destinations, and absolute deadlines

Slurm 18125 qualified runtime operations 100–111 against both the previous
platform reference and new C++23 controls. New controls passed normally and
with address/undefined-behavior sanitizers and leak detection.

100 creates a PollSet with explicit maximum registration count and a payload
budget of 24 bytes per entry. Transient poll descriptors and owner snapshots
have separately bounded native overhead proportional to that maximum.
101 inserts/updates a socket registration and caller tag, 102 removes it,
103 waits once against an absolute CLOCK_MONOTONIC deadline, and 104–106 read
the ordered ready event socket, flags, and tag. Flags are readable=1,
writable=2, error=4, hangup=8, invalid=16. A deadline of zero polls immediately;
no events returns zero without errno. Registration overflow leaves state intact.

Registrations retain logical IDs. Wait snapshots hold socket owners to prevent
fd reuse, release the registry lock before poll, and revalidate IDs afterward.
A closed registration yields invalid=16 with its original tag. Closing during
a wait can defer observation until another event or the deadline. PollSet
calls serialize internally; callers must also serialize a multi-call event
read against another wait or registration mutation.

107 sets an explicit IPv4 UDP destination without connect; 108 sends to it,
including zero-length datagrams. 109 returns the last consumed sender IPv4;
63 still returns its port. Truncation reports EMSGSIZE and preserves sender
metadata. Existing loopback UDP operations remain available.

110/111 perform one TCP send/receive against an unchanged absolute deadline.
Partial transfer counts are returned to Sounio loops. EINTR/EAGAIN retries
retain that deadline; timeout returns zero with ETIMEDOUT, and EOF returns
zero without errno. Zero-length stream operations complete immediately.
Algorithms accumulating send_all/read_exact progress remain in Sounio.

Controls cover budgets, bad handles/types/indices, registration order and tags,
closed IDs and non-reuse, a destination at 127.0.0.2, sender identity, empty and
truncated datagrams, interrupted absolute waits, stream transfer, timeout, EOF,
and existing platform behavior. This receipt qualifies the C ABI; Sounio API
qualification is a separate gate.

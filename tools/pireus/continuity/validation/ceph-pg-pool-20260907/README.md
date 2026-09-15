# Isolated PostgreSQL storage pool

The new `pireus_pg_ssd` pool has eight PGs, size3/min_size2, SSD placement
and three distinct hosts per PG. A flat per-pool weight set gives OSD0 zero
weight. Its 64 GiB logical quota bounds additional capacity consumption.
Existing pool mappings and existing CRUSH topology were compared before
and after and remained unchanged. No existing pool, OSD or data was removed.

Ceph19.2.3's upmap optimizer derives target weights from the CRUSH rule,
without using the pool-specific weight set. Therefore the automatic
balancer now explicitly includes the previously eligible existing pools
and excludes this new pool. It remains enabled in upmap mode. Existing
pools retain automatic balancing, but future pools require explicit
`ceph balancer pool add POOL` enrollment. Do not clear that allowlist while
this OSD0 exclusion is required; do not manually upmap this pool onto OSD0.
The pool autoscaler is off at eight PGs; other pools' settings are preserved.

A separate CSI identity is scoped to this pool. Its key and Secret manifest
remain private and are not in Git. The 64 GiB PVC uses Retain; the original
rehearsal PVC is also retained. The fresh-volume write/fsync and new-pod
read controls passed. Full restored-database I/O and timing are a separate
gate, currently running as V6. No global Ceph I/O repair is claimed:
the OSD0 warnings remain visible and its original workloads still need repair.

Creation: runtime/relocation_pool.py, executed on the Ceph admin host.
Provisioning: runtime/relocation_isolated_storage.py.
Placement/custody: summary.json, before.json, after.json, balancer-final.json.
Volume evidence: ../relocation-isolated-volume-20260907/.
The public files were selected explicitly; the sibling private credential
file under the host evidence directory was excluded.

References:
https://docs.ceph.com/en/squid/rados/operations/crush-map/
https://github.com/ceph/ceph/blob/v19.2.3/src/osd/OSDMap.cc
https://github.com/ceph/ceph/blob/v19.2.3/src/pybind/mgr/balancer/module.py

The native capacity metadata checker now runs in Node. Slurm18151 passed the current-source gate and31 source controls covering declaration/accessor tiers, IR sufficiency, stale/literal bounds, relocation and label sentinels, rc19/20/22/23 refusals, legacy ELF protection, ELF load-address preservation, return jumps, float register classification, string table bounds, and driver labels. Every input identity passed final verification. This is static source evidence; larger capacity and runtime emission still require separate native controls. The legacy Python checker remains archived and was not executed.

Slurm 18200 qualified additional register metadata checks and their rejection
controls: NC_REG_CLASS declaration/initialization, bounded get/set, dropped-write
counter, high-water reset, NC_HIGH_SEEN word count/reset/word guard, and complete
NC_IS_FLOAT_REG reset. These verify source coherence; they do not establish a
larger compiler capacity or qualify an expanded compiler artifact.

The real binary asset producer uses 32 MiB readonly data and a 192 MiB ELF buffer.
Slurm 18374 qualified those declaration/accessor defaults and all44 metadata
mutation controls against the sealed dc04 source. The checker now binds the
static_asset_limit accessor to a positive extent no larger than readonly data
capacity; altered, zero and missing limit accessors are rejected. Existing
overflow23 and ELF base-address guards remain in force. This is static source
coherence, separate from Slurm18362's actual readonly asset execution.

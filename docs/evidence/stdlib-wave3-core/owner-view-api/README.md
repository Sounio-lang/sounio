# Owner-bound byte and UTF8 views

ByteView and Utf8View expose checked spans over their original owners. Constructors
and subviews retain an ordinary compiler-tracked owner reference. Payload bytes
are borrowed without copying. The view/result objects can still allocate in the
current runtime; no allocation-free execution claim is made.

ByteView uses a half-open byte range [start,end) in ByteBuffer. byte_view_len
reports its length, byte_view_get returns ByteValue with an offset relative to
the view, and byte_view_slice takes another relative half-open range. Construction
propagates an owner failure and uses status4 for invalid bounds.

Utf8View uses the same byte-range convention over validated Utf8Text. Both ends
must be scalar boundaries (or the owner's byte length). Invalid boundaries return
status5; invalid bounds return4. byte_len and byte_at use relative bytes;
next returns a Utf8Scalar whose next offset is relative to the view. scalar_len
is explicit. Empty views and embedded NUL remain representable.

Both types have private fields. Public constructors establish their invariants;
external struct-literal forgery is refused. Views have no close operation because
they retain a borrow. The original owner is closed after the views' last use.
Subviews continue borrowing the original owner even after the parent view's last
use. The existing conservative match-loan rule can retain a result's owner loan
through an Err arm: cleanup is placed after the complete match in the witness.

Both collections::mod and collections::lib export ByteView, new, len, get and
slice. Both text::mod and text::lib export Utf8View, new, byte_len, byte_at, next,
scalar_len and slice. The composed native witness imports types through one
facade and functions through the other, so it tests actual public reexports.

## Prerequisites and exact source

The owner graph/ref-enum/tuple compiler changes qualified in26f079, the merged
preseed fix a7eb6b2ae7 and the receiver/array fixes4658b4b26e +4aecd9bb62 +040a99d903
are compiler prerequisites. This stdlib commit does not replace those compiler
commits. It uses the actual raw ELF from the separate source-built receiver lane.

The stdlib snapshot is core HEAD1ffa791541 plus the explicit view overlay. It
contains the original legacy HeapVec module with only the four qualified Mut
annotations, byte_buffer_budget from services commitd499506a4c (core adoption9cf),
and immutable BIDI2b82739c81 (core adoption1ffa, sourcec9fcb10f). The uncommitted
generic HeapVec draft is excluded from the snapshot and this API commit.

Raw compiler SHA256:
3c39904b0e811059211b429277112a4e02b507f36d4dff9dd8173360a3bfd784.
Compiler-source manifest SHA256:
c337959148524fc7ffbb942b23745178897fb84f32c13c69e7b55b29b5981471.
Platform ABI112 SHA256:
de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4.
Executed stdlib/source manifest SHA256:
87ae69767334bae076ceabe6c36c013e7f893a5eb0eaadb6c7e12ec2fea4c489.

The original owner_views.sio and core_generic_lib_facade.sio were compared
byte-for-byte to the previously qualified original-consumer snapshots. No
assertion, input, parameter or negative diagnostic was weakened. The original
owner-view positive covers A, NUL, e-acute and a supplementary emoji, checked
bounds, byte/scalar distinctions and subviews. The composed facade witness
additionally executes NUL through both public facade pairs.

## Native API gate19173

Slurm **19173** completed0:0 in29s on DL380. All29 declared phases passed their
exact expected status:

- Six leaf/facade module checks returned0.
- Three positives checked, compiled and ran natively at0 with exact markers:
  OWNER_VIEWS_OK, OWNER_VIEWS_FACADES_OK and CORE_GENERIC_LIB_FACADE_OK.
- Seven negatives each checked and compiled at1 without an ELF. Owner mutation,
  close, surviving-subview mutation and UTF8-owner close require E038. Escaping
  a local owner in a result requires E091. Forging either private view requires
  E259. E264 is forbidden as a substitute rejection.

The public gate pins compiler, platform, all stdlib/assets/wrappers, version
metadata, test inputs and executable bytes before/after. Its source archive
retains the primed wrapper metadata and actual input files. STATUS and the outer
worker both return0. The raw compiler is explicitly identified as a local source
build; none of these results is a committed-wrapper e33 qualification.

The raw19150 worker independently passes the returned-loan C++23 matrix
(51 checks,51 compilations,21 native cases and30 expected compile refusals),
seven other positive receivers including literal-array len, eight exact negative
receiver programs and the unchanged original HeapVec methods program natively.

## Explicit remaining limits

Raw19150's **aggregate status remains FAILED1**: the unchanged aliases_fields
control checks0, but its alias/field/reference lowering produces a body-less
i64_get and the unchanged native f4 guard refuses emission. That case is still
mandatory. The API worker verifies every other receiver result and retains this
specific refusal before running the selected view gate. A successful view gate
does not relabel that raw compiler globally accepted or generics complete.

HeapVec<T>/HeapDeque<T>, linear element disposal, typed storage/borrow hooks,
primitive source-string/C-string NUL/character ABI, coherent strict BIDI/CLDR and
the final zero-failure whole-stdlib result are separate requirements. In
particular, array/slice shape recognition for builtin len does not establish a
general native slice ABI. All original known-limit witnesses remain unchanged.

Failed API packaging runs and the earlier composed-facade rejection are retained
in receiver-dispatch01/view-api-failed-history.tar.gz, SHA256
169ee02fc7563fffdfbbaa03404b3a2f526171b7b44bf0d24b5464151ece0d7b.
They are not counted as API passes. The final source includes CHANGELOG.md and
CITATION.cff needed by the wrapper's version path and hashes both.

## Durable custody

The complete source, gate receipts, raw compiler with its explicit aggregate
failure and ABI112 runtime are preserved by Slurm19205 under:
/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/owner-view-api01

ARCHIVES.sha256 and ARCHIVES.final.verify bind the source, receipt, compiler and
runtime files. The exact hashes are retained alongside this document once the
archive job completes. The earlier receiver component/failure history remains
in its separately qualified durable receiver-dispatch01 directory.

## Final committed gate qualification19212

After removing one trailing blank line from the shell gate, Slurm **19212**
repeated the complete unchanged29-phase API matrix and completed0:0 in27s.
All implementation/probe bytes and the final shell gate were compared directly
with the staged worktree. WORKTREE_INPUTS.sha256 records that comparison set.
The exact shell gate SHA is
1e2de6e0b900c659b4f5a15bb6603e6850f8dbf3a30fc3add706d48bed1c05aa.
Final executed source-manifest SHA:
b6f93e39fe0d5f7cf6f90406612b097d333c88da0c57d6dadd14e29a9f7f1470.
The earlier19173 source/receipts remain unchanged.

Slurm19217 preserves the complete final source and receipts in the same durable
directory. FINAL_ARCHIVES.sha256 and FINAL_ARCHIVES.verify record their exact
hashes and recheck the previous19205 archives. Raw3c399, ABI112, original witness
identity, all exact negative diagnostics and the still-failed mandatory alias
control retain the limits described above.

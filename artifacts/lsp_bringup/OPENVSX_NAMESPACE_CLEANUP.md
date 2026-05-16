# OpenVSX namespace cleanup — `sounio` (dormant)

During the 2026-05-16 publication push, an extra namespace was created
on https://open-vsx.org named **`sounio`** (no suffix). It now holds
exactly one extension version: `sounio.sounio-vscode@1.1.0`, which is
in the registry's *inactive/pending* state and not visible in search
or download.

The intended namespace is `sounio-lang` (matches the GitHub org
`sounio-lang/sounio` from `package.json` `repository.url`), and the
live extension shipped there as `sounio-lang.sounio-vscode@1.1.1`.

## What to do

This dormant namespace is harmless — nothing routes to it, nothing
imports from it. You can leave it indefinitely. If you'd rather
retire it, two options:

### Option A: keep it but redirect

Log in at https://open-vsx.org as the namespace owner
(`agourakis82`). Mark the namespace deprecated with a pointer to
`sounio-lang/sounio-vscode`. This keeps any future search hit
directing users to the right place.

### Option B: full removal

OpenVSX does not currently support deleting an entire namespace via
the web UI. The Eclipse Foundation moderators can do it on request:

1. File an issue at https://github.com/eclipse/openvsx/issues
2. Subject: "Please retire namespace `sounio` (mistaken duplicate of
   `sounio-lang`)"
3. Body: include the publisher login (`agourakis82`), the
   accidentally-created namespace (`sounio`), and the intended
   namespace (`sounio-lang`), plus a note that the inactive extension
   `sounio.sounio-vscode@1.1.0` is the only entry.

Either way, no in-repo action is needed.

## Why it happened

The original `package.json` shipped with `"publisher": "sounio"`,
which `npx ovsx publish` will happily create as a new namespace when
the token is authorized. OpenVSX's automated verification flow tries
to link `publisher` ↔ `repository.url`'s GitHub org; mismatched names
(`sounio` vs `sounio-lang`) leave the publish in the inactive state.

The package.json publisher field has been corrected to `sounio-lang`
on commit `df303044`. Republishing under that name produces an
auto-active extension at https://open-vsx.org/extension/sounio-lang/sounio-vscode.

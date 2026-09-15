#!/usr/bin/env python3
"""Private schema, privilege and sequence checks on the isolated relocation target."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from relocation_dump import source, ident
from relocation_rehearsal import target, literal, ADMIN_DB, POD, kubectl

def run(command, output, error):
    with output.open("wb") as out, error.open("ab") as err:
        result = subprocess.run(command, stdout=out, stderr=err, timeout=180)
    if result.returncode:
        raise RuntimeError("Metadata command failed; private diagnostic retained")
    return output.read_bytes()

def query(builder, db, statement, output):
    return json.loads(run(builder("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c",
        "SELECT COALESCE(json_agg(t),'[]'::json) FROM (" + statement + ") t"]),
        output, output.with_suffix(".err")))

def normalize_dump(data):
    # Only psql's randomly generated safety token is nondeterministic.
    return re.sub(rb"(?m)^(\\(?:un)?restrict) [A-Za-z0-9]+$", rb"\1 TOKEN", data)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--archive-prefix", default="typed")
    args = ap.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    policy = json.loads(kubectl("get", "networkpolicy", "pireus-pg-relocation-rehearsal", "-o", "json"))
    if policy["spec"].get("ingress") or policy["spec"]["podSelector"] != {"matchLabels": {"app": "pireus-pg-relocation"}}:
        raise RuntimeError("Isolated target policy missing")
    started = time.monotonic()
    inventory = json.loads((args.source_output / "inventory.json").read_text())
    reports = []
    for i, entry in enumerate(inventory["databases"]):
        if not entry["datallowconn"]:
            continue
        db = entry["datname"]
        prefix = args.output / ("db-" + str(i))
        # Extract only metadata from the exact restored archive on its owning pod.
        archive_path = "/var/lib/postgresql/data/" + args.archive_prefix + "-db-" + str(i) + ".dump"
        archived = run(["kubectl", "-n", "beagle", "exec", POD, "--", "pg_restore",
                        "--schema-only", "--create", "-f", "-",
                        archive_path],
                       prefix.with_suffix(".archive-schema.sql"), prefix.with_suffix(".err"))
        live = run(target("pg_dump", db, ["--schema-only", "--create"]),
                   prefix.with_suffix(".target-schema.sql"), prefix.with_suffix(".err"))
        schema_equal = normalize_dump(archived) == normalize_dump(live)
        extensions = "SELECT extname,extversion,pg_get_userbyid(extowner) AS owner FROM pg_extension ORDER BY extname"
        ext_source = query(source, db, extensions, prefix.with_suffix(".source-extensions.json"))
        ext_target = query(target, db, extensions, prefix.with_suffix(".target-extensions.json"))
        seqs = query(source, db,
            "SELECT schemaname,sequencename FROM pg_sequences ORDER BY schemaname,sequencename",
            prefix.with_suffix(".sequence-names.json"))
        sequence_query = " UNION ALL ".join(
            "SELECT " + literal(s["schemaname"]) + " AS schema," + literal(s["sequencename"]) +
            " AS name,last_value::text AS last_value,is_called FROM " +
            ident(s["schemaname"]) + "." + ident(s["sequencename"]) for s in seqs)
        sequence_equal = True
        if seqs:
            source_seq = query(source, db, sequence_query, prefix.with_suffix(".source-sequences.json"))
            target_seq = query(target, db, sequence_query, prefix.with_suffix(".target-sequences.json"))
            sequence_equal = source_seq == target_seq
        expected = json.loads((args.source_output / ("db-" + str(i)) / "fingerprints.json").read_text())
        if "sequences" not in expected:
            raise RuntimeError("Complete source sequence capture missing")
        archive_sequence_equal = expected["sequences"] == (target_seq if seqs else [])
        report = {"database_index": i, "schema_equal": schema_equal,
                  "captured_sequence_states_equal": archive_sequence_equal,
                  "extension_owners_versions_equal": ext_source == ext_target,
                  "sequence_count": len(seqs), "live_sequence_states_equal": sequence_equal}
        reports.append(report)
        print(json.dumps(report), flush=True)
    queries = {
        "roles": "SELECT rolname,rolsuper,rolinherit,rolcreaterole,rolcreatedb,rolcanlogin,rolreplication,rolbypassrls,rolconnlimit,rolpassword,rolvaliduntil::text FROM pg_authid ORDER BY rolname",
        "memberships": "SELECT pg_get_userbyid(roleid) AS role,pg_get_userbyid(member) AS member,pg_get_userbyid(grantor) AS grantor,admin_option,inherit_option,set_option FROM pg_auth_members ORDER BY 1,2,3",
        "database_properties": "SELECT datname,pg_get_userbyid(datdba) AS owner,encoding,datlocprovider,datistemplate,datallowconn,datconnlimit,datcollate,datctype,daticulocale,daticurules,datcollversion,(SELECT json_agg(a ORDER BY a.grantee,a.grantor,a.privilege_type,a.is_grantable) FROM (SELECT pg_get_userbyid(grantee) AS grantee,pg_get_userbyid(grantor) AS grantor,privilege_type,is_grantable FROM aclexplode(datacl)) a) AS acl FROM pg_database WHERE datname<>" + literal(ADMIN_DB) + " ORDER BY datname",
        "role_database_settings": "SELECT COALESCE(d.datname,'ALL') AS db,COALESCE(r.rolname,'ALL') AS role,s.setconfig FROM pg_db_role_setting s LEFT JOIN pg_database d ON d.oid=s.setdatabase LEFT JOIN pg_roles r ON r.oid=s.setrole WHERE d.datname IS DISTINCT FROM " + literal(ADMIN_DB) + " ORDER BY 1,2",
    }
    comparisons = {}
    for label, statement in queries.items():
        expected = query(source, "memory", statement, args.output / (label + "-source-private.json"))
        actual = query(target, ADMIN_DB, statement, args.output / (label + "-target-private.json"))
        comparisons[label] = {"equal": expected == actual, "source_count": len(expected), "target_count": len(actual)}
    summary = {"elapsed_seconds": time.monotonic() - started, "databases": reports, "catalogs": comparisons,
               "source_write_pause": False, "sequence_snapshot_atomic": False,
               "production_cutover_accepted": False}
    summary["metadata_equal"] = all(all(r[k] for k in ("schema_equal", "extension_owners_versions_equal", "captured_sequence_states_equal")) for r in reports) and all(c["equal"] for c in comparisons.values())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("METADATA_REHEARSAL_COMPLETE " + json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()

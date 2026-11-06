#!/usr/bin/env python3
"""Timed isolated rehearsal. Never changes the source endpoint or pauses writers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time
from relocation_dump import fingerprint, ident, SOURCE_POD

POD = "pireus-pg-relocation-0"
ADMIN_DB = "pireus_migration_admin_20260907"

def kubectl(*args):
    return subprocess.check_output(["kubectl", "-n", "beagle", *args], timeout=40)

def target(tool, db, extra=(), interactive=False):
    cmd = ["kubectl", "-n", "beagle", "exec"]
    if interactive:
        cmd.append("-i")
    return cmd + [POD, "--", "sh", "-c",
                  'tool="$1"; db="$2"; shift 2; exec "$tool" -U "$POSTGRES_USER" -d "$db" "$@"',
                  "--", tool, db, *extra]

def literal(value):
    return "'" + value.replace("'", "''") + "'"

def wait_file(path, deadline, producer=None):
    while not path.exists():
        if producer is not None and producer.poll() is not None:
            raise RuntimeError("Source backup exited before publishing required artifact; private diagnostic retained")
        if time.monotonic() > deadline:
            raise RuntimeError("Rehearsal artifact deadline exceeded")
        time.sleep(1)

def sql(statement, error_path):
    with error_path.open("ab") as errors:
        p = subprocess.run(target("psql", ADMIN_DB, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-f", "-"], True),
                           input=statement.encode(), stdout=subprocess.PIPE, stderr=errors, timeout=60)
    if p.returncode:
        raise RuntimeError("Target SQL failed; private diagnostic retained")
    return p.stdout

def compare(entry, source_dir, output, deadline, producer=None):
    wait_file(source_dir / "fingerprints.json", deadline, producer)
    expected = json.loads((source_dir / "fingerprints.json").read_text())
    if expected["record_format"] != "postgres-record-sha256-v2":
        raise RuntimeError("Unexpected fingerprint representation")
    db = entry["datname"]
    error_path = output / (entry["index"] + "-verify-private.log")
    start = time.monotonic()
    with error_path.open("ab") as errors:
        holder = subprocess.Popen(target("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1"], True),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
        try:
            holder.stdin.write(b"BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; SELECT pg_export_snapshot();\n")
            holder.stdin.flush()
            if not select.select([holder.stdout], [], [], 60)[0]:
                raise RuntimeError("Target snapshot timeout")
            snap = holder.stdout.readline().decode().strip()
            if not re.fullmatch(r"[0-9A-Fa-f]+-[0-9A-Fa-f]+-[0-9]+", snap):
                raise RuntimeError("Invalid target snapshot")
            mismatches = []
            for table in expected["tables"]:
                if table["schema"] == "pgivm" and table["name"] == "pg_ivm_immv" and table["rows"]:
                    raise RuntimeError("Existing IMMV metadata is unsupported")
                order = ", ".join("t." + ident(k) for k in table["primary_key"]) or 't::text COLLATE "C"'
                query = ("SELECT encode(sha256(convert_to(t::text,'UTF8')),'hex') FROM ONLY " + ident(table["schema"]) + "." +
                         ident(table["name"]) + " t ORDER BY " + order)
                actual = fingerprint(db, query, snap, error_path, target, bool(table["primary_key"]))
                if any(actual[k] != table[k] for k in ["rows", "sha256"]):
                    mismatches.append({"expected": table, "actual": actual})
            for key, query in [
                ("large_object_pages", "SELECT loid,pageno,encode(data,'hex') FROM pg_largeobject ORDER BY loid,pageno"),
                ("large_object_metadata", "SELECT oid,pg_get_userbyid(lomowner),lomacl::text FROM pg_largeobject_metadata ORDER BY oid"),
            ]:
                actual = fingerprint(db, query, snap, error_path, target)
                if any(actual[k] != expected[key][k] for k in ["rows", "sha256"]):
                    mismatches.append({"kind": key, "actual": actual})
            report = {"database_index": entry["index"], "tables": len(expected["tables"]),
                      "mismatches": mismatches, "seconds": time.monotonic() - start}
            (output / (entry["index"] + "-comparison-private.json")).write_text(json.dumps(report, indent=2))
            holder.stdin.write(b"ROLLBACK;\n\\q\n")
            holder.stdin.flush()
            holder.stdin.close()
            holder.wait(timeout=30)
            if mismatches:
                raise RuntimeError("Typed data parity failed")
            return {"tables": len(expected["tables"]), "seconds": report["seconds"], "mismatches": 0}
        finally:
            if holder.poll() is None:
                holder.terminate()
                try:
                    holder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    holder.kill()
                    holder.wait()

def restore_database_settings(entry, source_directory, log_path):
    inventory = json.loads((source_directory / "inventory.json").read_text())
    if "database_settings" not in inventory:
        raise RuntimeError("Captured source database settings missing")
    db = ident(entry["datname"])
    statements = ["BEGIN", "ALTER DATABASE " + db + " RESET ALL"]
    for item in inventory["database_settings"]:
        if item["database"] == entry["datname"]:
            for setting in item["setconfig"]:
                key, value = setting.split("=", 1)
                statements += ["SELECT pg_catalog.set_config(" + literal(key) + "," + literal(value) + ",false)",
                               "ALTER DATABASE " + db + " SET " + ident(key) + " FROM CURRENT"]
    statements.append("COMMIT")
    sql(";\n".join(statements) + ";\n", log_path)

def restore(entry, source, output, deadline, producer=None):
    directory = source / entry["index"]
    # partial fingerprints only appear AFTER pg_dump has completed and closed the archive.
    wait_file(directory / "fingerprints.partial.json", deadline, producer)
    partial = json.loads((directory / "fingerprints.partial.json").read_text())
    archive = directory / "database.dump"
    if archive.stat().st_size != partial["dump_bytes"]:
        raise RuntimeError("Archive not finalized")
    remote = "/var/lib/postgresql/data/" + output.name + "-" + entry["index"] + ".dump"
    log_path = output / (entry["index"] + "-restore-private.log")
    start = time.monotonic()
    with archive.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with archive.open("rb") as inp, log_path.open("ab") as log:
        p = subprocess.run(["kubectl", "-n", "beagle", "exec", "-i", POD, "--", "sh", "-c",
                            'set -e; umask 077; test ! -e "$1"; cat > "$1"', "--", remote],
                           stdin=inp, stdout=log, stderr=log, timeout=600)
    if p.returncode:
        raise RuntimeError("Archive transfer failed")
    remote_digest = kubectl("exec", POD, "--", "sha256sum", remote).decode().split()[0]
    if digest != remote_digest:
        raise RuntimeError("Archive custody mismatch")
    report = {"database_index": entry["index"], "transfer_seconds": time.monotonic() - start}
    db = entry["datname"]
    bulk = entry.get("bulk_prepared", False)
    if not bulk:
        exists = sql("SELECT count(*) FROM pg_database WHERE datname=" + literal(db), log_path).strip()
        if exists == b"1":
            sql("ALTER DATABASE " + ident(db) + " ALLOW_CONNECTIONS false; "
                "ALTER DATABASE " + ident(db) + " IS_TEMPLATE false; "
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=" +
                literal(db) + " AND pid<>pg_backend_pid();", log_path)
    start = time.monotonic()
    with log_path.open("ab") as log:
        if bulk:
            check = subprocess.run(target("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c",
                "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE c.relkind IN ('r','m') AND n.nspname !~ '^pg_' AND n.nspname<>'information_schema'"]),
                stdout=subprocess.PIPE, stderr=log, timeout=30)
            if check.returncode or check.stdout.strip() != b"0":
                raise RuntimeError("Bulk destination was not prepared empty")
            # CREATE TABLE and COPY share one transaction so minimal-WAL bulk loading applies.
            p = subprocess.run(target("pg_restore", db,
                ["--single-transaction", "--exit-on-error", "--section=pre-data", "--section=data", remote]),
                stdout=log, stderr=log, timeout=1800)
            if p.returncode == 0:
                p = subprocess.run(target("pg_restore", db,
                    ["--jobs=4", "--exit-on-error", "--section=post-data", remote]),
                    stdout=log, stderr=log, timeout=1800)
        else:
            p = subprocess.run(target("pg_restore", ADMIN_DB,
                                      ["--jobs=4", "--exit-on-error", "--create", "--clean", "--if-exists", remote]),
                               stdout=log, stderr=log, timeout=1800)
    report.update({"restore_seconds": time.monotonic() - start, "restore_rc": p.returncode})
    (output / (entry["index"] + "-restore.json")).write_text(json.dumps(report, indent=2))
    if p.returncode:
        raise RuntimeError("Restore failed; private diagnostic retained")
    wait_file(directory / "fingerprints.json", deadline, producer)
    expected = json.loads((directory / "fingerprints.json").read_text())
    if "sequences" not in expected:
        raise RuntimeError("Complete sequence capture missing")
    sequence_sql = "\n".join("SELECT pg_catalog.setval(" +
        literal(ident(item["schema"]) + "." + ident(item["name"])) + "::regclass," +
        str(int(item["last_value"])) + "," + str(item["is_called"]).lower() + ");"
        for item in expected["sequences"])
    with log_path.open("ab") as log:
        result = subprocess.run(target("psql", entry["datname"],
            ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-f", "-"], True),
            input=sequence_sql.encode(), stdout=log, stderr=log, timeout=60)
    if result.returncode:
        raise RuntimeError("Complete sequence restore failed")
    report["sequences_restored"] = len(expected["sequences"])
    restore_database_settings(entry, source, log_path)
    report["comparison"] = compare(entry, directory, output, deadline, producer)
    print(json.dumps(report), flush=True)
    return report

def verify_final_source(token):
    result = subprocess.run(["kubectl", "-n", "beagle", "exec", SOURCE_POD, "--",
        "nsenter", "-t", "1", "-m", "-p", "-n", "--", "python3",
        "/var/lib/pireus/pg-relocation-20260907/controller.py", "status", "--token", token],
        capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Final source authority/fence observation failed")
    state = json.loads(result.stdout)
    if state["state"] != "FENCED" or state["remaining_seconds"] <= 0:
        raise RuntimeError("Final source is no longer fenced within its deadline")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--globals-already-restored", action="store_true")
    parser.add_argument("--bulk-prepared", action="store_true")
    parser.add_argument("--final-cutover-token")
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    pod = json.loads(kubectl("get", "pod", POD, "-o", "json"))
    env = {e["name"]: e.get("value") for e in pod["spec"]["containers"][0]["env"]}
    if pod["spec"]["nodeName"] != "r770-proxmox" or env["PGDATA"] != "/var/lib/postgresql/data/pgdata-source-bootstrap":
        raise RuntimeError("Wrong isolated target")
    policy = json.loads(kubectl("get", "networkpolicy", "pireus-pg-relocation-rehearsal", "-o", "json"))
    if policy["spec"].get("ingress") or policy["spec"]["podSelector"] != {"matchLabels": {"app": "pireus-pg-relocation"}}:
        raise RuntimeError("Target ingress isolation missing")
    if json.loads(kubectl("get", "ciliumnetworkpolicy", "-o", "json"))["items"]:
        raise RuntimeError("Unexpected additional target ingress policy")
    if args.bulk_prepared:
        with (args.output / "bulk-settings-private.log").open("ab") as errors:
            settings = subprocess.run(target("psql", ADMIN_DB, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c",
                "SELECT current_setting('wal_level'),current_setting('max_wal_senders'),current_setting('archive_mode'),current_setting('fsync'),current_setting('full_page_writes'),current_setting('synchronous_commit'),current_setting('cron.launch_active_jobs')"]),
                stdout=subprocess.PIPE, stderr=errors, timeout=30)
        if settings.returncode or settings.stdout.strip() != b"minimal|0|off|on|on|on|off":
            raise RuntimeError("Bulk WAL/durability settings do not match")
    if args.final_cutover_token:
        verify_final_source(args.final_cutover_token)
    start = time.monotonic()
    deadline = start + 1800
    with (args.output / "source-private.log").open("xb") as log:
        source_process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("relocation_dump.py")),
                                          "--output", str(args.source_output)], stdout=log, stderr=log)
        try:
            wait_file(args.source_output / "db-0" / "database.dump", deadline, source_process)
            inventory = json.loads((args.source_output / "inventory.json").read_text())
            # Retain the original bootstrap role identity (OID10), then restore its attributes/password.
            globals_sql = (args.source_output / "globals-private.sql").read_text()
            create = "CREATE ROLE memory;"
            if globals_sql.splitlines().count(create) != 1:
                raise RuntimeError("Unexpected bootstrap role dump; do not guess")
            filtered = "\n".join("-- Existing source-aligned bootstrap role." if line == create else line
                                 for line in globals_sql.splitlines()) + "\n"
            (args.output / "globals-filtered-private.sql").write_text(filtered)
            if not args.globals_already_restored:
                sql("BEGIN;\n" + filtered + "\nCOMMIT;\n", args.output / "globals-private.log")
            entries = [dict(db, index="db-" + str(i), bulk_prepared=args.bulk_prepared) for i, db in enumerate(inventory["databases"])
                       if db["datallowconn"]]
            # CREATE DATABASE and concurrent restores contend on shared checkpoints.
            # Backups still overlap this sequential destination restore pipeline.
            reports = [restore(e, args.source_output, args.output, deadline, source_process) for e in entries]
            if source_process.wait(timeout=60):
                raise RuntimeError("Source backup failed")
            with (args.output / "metadata-private.log").open("xb") as metadata_log:
                subprocess.run([sys.executable, str(Path(__file__).with_name("relocation_metadata.py")),
                    "--source-output", str(args.source_output), "--output", str(args.output / "metadata"),
                    "--archive-prefix", args.output.name], stdout=metadata_log, stderr=metadata_log,
                    check=True, timeout=300)
            metadata = json.loads((args.output / "metadata" / "summary.json").read_text())
            if args.final_cutover_token:
                verify_final_source(args.final_cutover_token)
            elapsed = time.monotonic() - start
            summary = {"elapsed_seconds": elapsed, "overhead_allowance_seconds": 120,
                       "core_within_780_seconds": elapsed <= 780,
                       "database_count": len(reports), "tables_compared": sum(r["comparison"]["tables"] for r in reports),
                       "mismatches": sum(r["comparison"]["mismatches"] for r in reports),
                       "record_format": "postgres-record-sha256-v2", "source_write_pause": bool(args.final_cutover_token),
                       "schema_acl_sequence_acceptance": metadata["metadata_equal"],
                       "application_acceptance": False, "bulk_prepared": args.bulk_prepared,
                       "final_runtime_mode_restored": not args.bulk_prepared, "reports": reports}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print("TIMED_TYPED_REHEARSAL_COMPLETE " + json.dumps(summary), flush=True)
        finally:
            if source_process.poll() is None:
                source_process.terminate()
                try:
                    source_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    source_process.kill()
                    source_process.wait()

if __name__ == "__main__":
    main()

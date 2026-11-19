#!/usr/bin/env python3
"""Private, per-database snapshot backup and COPY fingerprints; no source writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

SOURCE_POD = "pireus-spark-host-fence-f7sl5"
SOURCE_CONTAINER = "beagle-memory-pg-pdb"

def write_json(path, value):
    """Publish complete metadata atomically to concurrent rehearsal readers."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def source(tool, db, extra=(), interactive=False):
    cmd = ["kubectl", "-n", "beagle", "exec"]
    if interactive:
        cmd.append("-i")
    cmd += [SOURCE_POD, "--", "nsenter", "-t", "1", "-m", "-p", "--",
            "docker", "exec"]
    if interactive:
        cmd.append("-i")
    cmd += ["-u", "postgres", SOURCE_CONTAINER, "sh", "-c",
            'tool="$1"; db="$2"; shift 2; exec "$tool" -U "$POSTGRES_USER" -d "$db" "$@"',
            "--", tool, db, *extra]
    return cmd

def rows(db, query):
    query = "SELECT COALESCE(json_agg(t),'[]'::json) FROM (" + query + ") t"
    result = subprocess.run(source("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", query]),
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Source catalog query failed")
    return json.loads(result.stdout)

def ident(s):
    return '"' + s.replace('"', '""') + '"'

def fingerprint(db, query, snapshot, error_file, command_builder=source, prefer_index=False):
    sql = ("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; "
           "SET TRANSACTION SNAPSHOT '" + snapshot + "'; "
           "SET LOCAL statement_timeout='300s'; SET LOCAL lock_timeout='5s'; "
           "SET LOCAL TimeZone='UTC'; SET LOCAL DateStyle='ISO,YMD'; "
           "SET LOCAL IntervalStyle='postgres'; SET LOCAL extra_float_digits=3; "
           "SET LOCAL bytea_output='hex'; SET LOCAL search_path=pg_catalog; "
           + ("SET LOCAL enable_seqscan=off; SET LOCAL enable_bitmapscan=off; " if prefer_index else "")
           + "COPY (" + query + ") TO STDOUT; COMMIT;")
    cmd = ["timeout", "360", *command_builder("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", sql])]
    digest = hashlib.sha256()
    count = 0
    started = time.monotonic()
    with error_file.open("ab") as errors:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors)
        try:
            while chunk := p.stdout.read(1024 * 1024):
                digest.update(chunk)
                count += chunk.count(b"\n")
            rc = p.wait(timeout=15)
        except BaseException:
            p.kill()
            p.wait()
            raise
    if rc:
        raise RuntimeError("Snapshot COPY failed; see private diagnostic")
    return {"rows": count, "sha256": digest.hexdigest(), "seconds": time.monotonic() - started}

def dump_database(db, directory, command_builder=source):
    directory.mkdir(mode=0o700)
    error_file = directory / "private-errors.log"
    start = time.monotonic()
    with error_file.open("ab") as errors:
        holder = subprocess.Popen(
            command_builder("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1"], interactive=True),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
        try:
            holder.stdin.write(b"BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\nSELECT pg_export_snapshot();\n")
            holder.stdin.flush()
            # A bounded readiness wait prevents a disconnected transport hanging the lane.
            import select
            ready, _, _ = select.select([holder.stdout], [], [], 60)
            if not ready:
                raise RuntimeError("Snapshot exporter did not respond")
            snapshot = holder.stdout.readline().decode().strip()
            if not re.fullmatch(r"[0-9A-Fa-f]+-[0-9A-Fa-f]+-[0-9]+", snapshot):
                raise RuntimeError("Invalid exported snapshot")
            archive = directory / "database.dump"
            dump_start = time.monotonic()
            with archive.open("xb") as output:
                result = subprocess.run(
                    command_builder("pg_dump", db, ["-Fc", "--create", "--compress=zstd:1", "--snapshot=" + snapshot]),
                    stdout=output, stderr=errors, timeout=900)
            if result.returncode:
                raise RuntimeError("pg_dump failed; retain partial archive")
            report = {"database": db, "snapshot": snapshot,
                      "dump_seconds": time.monotonic() - dump_start,
                      "dump_bytes": archive.stat().st_size, "record_format": "postgres-record-sha256-v2", "tables": []}
            # Enumerate relations using the SAME exported snapshot.
            query = ("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY; "
                     "SET TRANSACTION SNAPSHOT '" + snapshot + "'; "
                     "SELECT COALESCE(json_agg(t),'[]'::json) FROM ("
                     "SELECT n.nspname AS schema,c.relname AS name,c.relkind, "
                     "COALESCE((SELECT json_agg(a.attname ORDER BY k.ord) "
                     "FROM pg_index i CROSS JOIN LATERAL unnest(i.indkey::smallint[]) "
                     "WITH ORDINALITY AS k(attnum,ord) JOIN pg_attribute a "
                     "ON a.attrelid=c.oid AND a.attnum=k.attnum "
                     "WHERE i.indrelid=c.oid AND i.indisprimary AND k.ord<=i.indnkeyatts),'[]'::json) AS primary_key "
                     "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                     "WHERE c.relkind IN ('r','m') AND n.nspname !~ '^pg_' "
                     "AND n.nspname <> 'information_schema' ORDER BY n.nspname,c.relname) t; COMMIT;")
            result = subprocess.run(command_builder("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", query]),
                                    stdout=subprocess.PIPE, stderr=errors, timeout=60)
            if result.returncode:
                raise RuntimeError("Snapshot relation inventory failed")
            tables = json.loads(result.stdout)
            for table in tables:
                relation = ident(table["schema"]) + "." + ident(table["name"])
                order = ", ".join("t." + ident(k) for k in table["primary_key"])
                if not order:
                    order = 't::text COLLATE "C"'
                query = "SELECT encode(sha256(convert_to(t::text,'UTF8')),'hex') FROM ONLY " + relation + ' t ORDER BY ' + order
                table.update(fingerprint(db, query, snapshot, error_file,
                                         command_builder=command_builder, prefer_index=bool(table["primary_key"])))
                if table["schema"] == "pgivm" and table["name"] == "pg_ivm_immv" and table["rows"]:
                    raise RuntimeError("Snapshot contains unsupported IVM metadata")
                report["tables"].append(table)
                write_json(directory / "fingerprints.partial.json", report)
            # Sequences are not MVCC data, and extension-owned sequences may be
            # omitted by pg_dump. Capture every sequence explicitly after the dump.
            sequence_names_sql = "SELECT COALESCE(json_agg(t),'[]'::json) FROM (SELECT schemaname AS schema,sequencename AS name FROM pg_sequences ORDER BY schemaname,sequencename) t"
            result = subprocess.run(command_builder("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", sequence_names_sql]),
                                    stdout=subprocess.PIPE, stderr=errors, timeout=60)
            if result.returncode:
                raise RuntimeError("Sequence inventory failed")
            sequence_names = json.loads(result.stdout)
            report["sequences"] = []
            if sequence_names:
                def sql_literal(value):
                    return "'" + value.replace("'", "''") + "'"
                sequence_query = " UNION ALL ".join(
                    "SELECT " + sql_literal(item["schema"]) + " AS schema," + sql_literal(item["name"]) +
                    " AS name,last_value::text AS last_value,is_called FROM " +
                    ident(item["schema"]) + "." + ident(item["name"]) for item in sequence_names)
                result = subprocess.run(command_builder("psql", db, ["-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c",
                    "SELECT COALESCE(json_agg(t),'[]'::json) FROM (" + sequence_query + ") t"]),
                    stdout=subprocess.PIPE, stderr=errors, timeout=60)
                if result.returncode:
                    raise RuntimeError("Sequence state capture failed")
                report["sequences"] = json.loads(result.stdout)
            report["sequence_capture_atomic_with_snapshot"] = False
            report["large_object_pages"] = fingerprint(
                db, "SELECT loid,pageno,encode(data,'hex') FROM pg_largeobject ORDER BY loid,pageno",
                snapshot, error_file, command_builder)
            report["large_object_metadata"] = fingerprint(
                db, "SELECT oid,pg_get_userbyid(lomowner),lomacl::text FROM pg_largeobject_metadata ORDER BY oid",
                snapshot, error_file, command_builder)
            report["total_seconds"] = time.monotonic() - start
            with archive.open("rb") as f:
                report["archive_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
            write_json(directory / "fingerprints.json", report)
            holder.stdin.write(b"ROLLBACK;\n\\q\n")
            holder.stdin.flush()
            holder.stdin.close()
            if holder.wait(timeout=30):
                raise RuntimeError("Snapshot exporter failed on close")
            return report
        finally:
            if holder.poll() is None:
                holder.terminate()
                try:
                    holder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    holder.kill()
                    holder.wait()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    databases = rows("memory", "SELECT datname,datistemplate,datallowconn,pg_database_size(oid) AS bytes "
                     "FROM pg_database ORDER BY datname")
    if rows("memory", "SELECT spcname FROM pg_tablespace WHERE spcname NOT IN ('pg_default','pg_global')"):
        raise RuntimeError("Nondefault tablespaces need an explicit destination map")
    settings = rows("memory", "SELECT d.datname AS database,s.setconfig FROM pg_db_role_setting s JOIN pg_database d ON d.oid=s.setdatabase WHERE s.setrole=0 ORDER BY d.datname")
    inventory = {"databases": databases, "source": SOURCE_POD, "reports": [], "database_settings": settings}
    write_json(args.output / "inventory.json", inventory)
    # Credentials remain in a0600 globals file; never print or commit its contents.
    cmd = ["kubectl", "-n", "beagle", "exec", SOURCE_POD, "--", "nsenter", "-t", "1", "-m", "-p",
           "--", "docker", "exec", "-u", "postgres", SOURCE_CONTAINER, "sh", "-c",
           'exec pg_dumpall -U "$POSTGRES_USER" --globals-only']
    with (args.output / "globals-private.sql").open("xb") as output, (args.output / "globals-private.err").open("xb") as errors:
        subprocess.run(cmd, stdout=output, stderr=errors, check=True, timeout=60)
    start = time.monotonic()
    for index, db in enumerate(databases):
        if not db["datallowconn"]:
            continue
        # This restore path has no support for existing pg_ivm metadata.
        extensions = rows(db["datname"], "SELECT extname FROM pg_extension")
        if any(e["extname"] == "pg_ivm" for e in extensions):
            count = rows(db["datname"], "SELECT count(*) AS count FROM pgivm.pg_ivm_immv")[0]["count"]
            if count:
                raise RuntimeError("Existing IMMV detected; stop before unsupported restore")
        report = dump_database(db["datname"], args.output / ("db-" + str(index)))
        inventory["reports"].append({"directory": "db-" + str(index), "database": db["datname"],
                                      "dump_seconds": report["dump_seconds"], "total_seconds": report["total_seconds"]})
        write_json(args.output / "inventory.json", inventory)
        print(json.dumps({"database_index": index, "dump_bytes": report["dump_bytes"],
                          "tables": len(report["tables"]), "seconds": report["total_seconds"]}), flush=True)
    inventory["backup_and_fingerprint_seconds"] = time.monotonic() - start
    inventory["per_database_snapshots"] = True
    inventory["cross_database_atomic_snapshot"] = False
    inventory["sequence_parity_verified"] = False
    write_json(args.output / "inventory.json", inventory)
    print("SOURCE_SNAPSHOT_BACKUP_COMPLETE", flush=True)

if __name__ == "__main__":
    main()

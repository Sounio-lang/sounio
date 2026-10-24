#!/usr/bin/env python3
"""Audit closed template0 through owned temporary clones; never open the original."""
import argparse, datetime, json, os, pathlib, subprocess, time, uuid
from relocation_dump import source, dump_database, ident
from relocation_rehearsal import target, literal, ADMIN_DB, kubectl
from relocation_metadata import normalize_dump
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args();os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    if json.loads(kubectl("get","ciliumnetworkpolicy","-o","json"))["items"]:
        raise RuntimeError("Target is not isolated")
    token=uuid.uuid4().hex;name="pireus_template0_audit_"+token
    marker="pireus-template0-audit:"+token
    log_path=args.output/"audit-private.log";owned=[];captures=[];schemas=[];properties=[]
    def execute(builder,db,sql):
        with log_path.open("ab") as log:
            p=subprocess.run(builder("psql",db,["-X","-qAt","-v","ON_ERROR_STOP=1","-c",sql]),
                             stdout=subprocess.PIPE,stderr=log,timeout=120)
        if p.returncode:raise RuntimeError("Audit SQL failed; private diagnostic retained")
        return p.stdout.strip()
    started=time.monotonic()
    report={"source_template0_opened":False,"source_application_write_pause":False,
            "temporary_admin_databases_created":0,"temporary_admin_databases_removed":0,
            "production_cutover_accepted":False,"template0_content_audit_pass":False}
    try:
        for label,builder,admin in [("source",source,"postgres"),("target",target,ADMIN_DB)]:
            if execute(builder,admin,"SELECT datallowconn,datistemplate FROM pg_database WHERE datname='template0'")!=b"f|t":
                raise RuntimeError("Original template0 is not closed")
            if execute(builder,admin,"SELECT count(*) FROM pg_database WHERE datname="+literal(name))!=b"0":
                raise RuntimeError("Audit database already exists")
            execute(builder,admin,"CREATE DATABASE "+ident(name)+" TEMPLATE template0")
            oid=execute(builder,admin,"SELECT oid FROM pg_database WHERE datname="+literal(name)).decode()
            if not oid.isdecimal():raise RuntimeError("Audit ownership identity unavailable")
            owned.append((label,builder,admin,oid))
            report["temporary_admin_databases_created"]+=1
            (args.output/"owned-databases-private.json").write_text(json.dumps(
                [{"side":x[0],"database":name,"oid":x[3],"token":token} for x in owned],indent=2))
            execute(builder,admin,"COMMENT ON DATABASE "+ident(name)+" IS "+literal(marker))
            properties.append(execute(builder,admin,"SELECT encoding,datlocprovider,datcollate,datctype,daticulocale,daticurules,datcollversion FROM pg_database WHERE datname="+literal(name)))
            captured=dump_database(name,args.output/label,command_builder=builder);captures.append(captured)
            with log_path.open("ab") as log:
                schema=subprocess.run(builder("pg_dump",name,["--schema-only","--create"]),
                                      stdout=subprocess.PIPE,stderr=log,check=True,timeout=120).stdout
            (args.output/(label+"-schema-private.sql")).write_bytes(schema);schemas.append(normalize_dump(schema))
            if execute(builder,admin,"SELECT datallowconn,datistemplate FROM pg_database WHERE datname='template0'")!=b"f|t":
                raise RuntimeError("Original template0 properties changed")
        def content(c):
            return {"record_format":c["record_format"],
                "tables":[{k:t[k] for k in ["schema","name","relkind","primary_key","rows","sha256"]} for t in c["tables"]],
                "sequences":c["sequences"],
                "large_object_pages":{k:c["large_object_pages"][k] for k in ["rows","sha256"]},
                "large_object_metadata":{k:c["large_object_metadata"][k] for k in ["rows","sha256"]}}
        report.update({"schema_equal":schemas[0]==schemas[1],"properties_equal":properties[0]==properties[1],
                       "typed_content_equal":content(captures[0])==content(captures[1]),
                       "user_tables":len(captures[0]["tables"]),"sequences":len(captures[0]["sequences"]),
                       "large_object_pages":captures[0]["large_object_pages"]["rows"]})
        report["template0_content_audit_pass"]=all(report[k] for k in ["schema_equal","properties_equal","typed_content_equal"])
        if not report["template0_content_audit_pass"]:raise RuntimeError("Template0 differs; preserve audit archives and stop migration")
    finally:
        cleanup_errors=[]
        for label,builder,admin,oid in reversed(owned):
            try:
                current=execute(builder,admin,"SELECT oid FROM pg_database WHERE datname="+literal(name)).decode()
                if current!=oid:raise RuntimeError("Audit database identity changed; preserve it")
                # The UUID and recorded OID identify only the database created by this run.
                execute(builder,admin,"DROP DATABASE "+ident(name)+" WITH (FORCE)")
                report["temporary_admin_databases_removed"]+=1
            except Exception:
                cleanup_errors.append(label)
        report["cleanup_pass"]=not cleanup_errors and len(owned)==report["temporary_admin_databases_removed"]
        report["cleanup_failed_sides"]=cleanup_errors
        report["elapsed_seconds"]=time.monotonic()-started
        report["template0_content_audit_pass"] &= report["cleanup_pass"]
        (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print("CLOSED_TEMPLATE_AUDIT "+json.dumps(report),flush=True)
    if not report["template0_content_audit_pass"]:raise SystemExit(2)
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Fresh read-only source-shape guards for the bounded logical migration."""
import argparse, datetime, json, os, pathlib
from relocation_dump import rows
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args();os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    cluster=rows("postgres", """SELECT
      pg_is_in_recovery() AS standby,
      current_setting('wal_level') AS wal_level,
      (SELECT count(*) FROM pg_prepared_xacts) AS prepared_transactions,
      (SELECT count(*) FROM pg_replication_slots) AS replication_slots,
      (SELECT count(*) FROM pg_replication_origin) AS replication_origins,
      (SELECT count(*) FROM pg_stat_replication) AS replication_clients,
      (SELECT count(*) FROM pg_tablespace WHERE spcname NOT IN ('pg_default','pg_global')) AS custom_tablespaces""")[0]
    dbs=rows("postgres","SELECT datname,datallowconn FROM pg_database ORDER BY datname")
    reports=[]
    for index,db in enumerate(dbs):
        if not db["datallowconn"]:continue
        counts=rows(db["datname"],"""SELECT
          (SELECT count(*) FROM pg_foreign_table) AS foreign_tables,
          (SELECT count(*) FROM pg_subscription) AS subscriptions,
          (SELECT count(*) FROM pg_publication) AS publications,
          (SELECT count(*) FROM pg_database WHERE datname=current_database() AND
            datcollversion IS DISTINCT FROM pg_database_collation_actual_version(oid)) AS database_collation_mismatches,
          (SELECT count(*) FROM pg_collation WHERE collversion IS NOT NULL AND
            collversion IS DISTINCT FROM pg_collation_actual_version(oid)) AS collation_mismatches,
          to_regclass('pgivm.pg_ivm_immv') IS NOT NULL AS ivm_present,
          to_regclass('cron.job') IS NOT NULL AS cron_present""")[0]
        counts["existing_immvs"]=rows(db["datname"],"SELECT count(*) AS n FROM pgivm.pg_ivm_immv")[0]["n"] if counts.pop("ivm_present") else 0
        counts["cron_jobs"]=rows(db["datname"],"SELECT count(*) AS n FROM cron.job")[0]["n"] if counts.pop("cron_present") else 0
        reports.append({"database_index":index,**counts})
    (args.output/"database-inventory-private.json").write_text(json.dumps(dbs,indent=2))
    accepted=([d["datname"] for d in dbs]==["memory","memory_test","paradedb","postgres","template0","template1"] and not cluster["standby"] and cluster["wal_level"]=="replica" and
              all(cluster[k]==0 for k in ["prepared_transactions","replication_slots","replication_origins","replication_clients","custom_tablespaces"]) and
              all(all(value==0 for key,value in d.items() if key!="database_index") for d in reports) and
              [d["datname"] for d in dbs if not d["datallowconn"]]==["template0"])
    summary={"observed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "source_shape_pass":accepted,"cluster":cluster,"databases":reports,
             "closed_template_content_audited":False,"source_write_pause":False,
             "production_cutover_accepted":False}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print("SOURCE_SHAPE_GUARD "+json.dumps(summary),flush=True)
    if not accepted:raise SystemExit(2)
if __name__=="__main__":main()

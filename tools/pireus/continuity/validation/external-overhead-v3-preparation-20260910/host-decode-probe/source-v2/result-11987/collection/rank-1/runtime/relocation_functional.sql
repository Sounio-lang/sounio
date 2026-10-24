BEGIN;
SET LOCAL statement_timeout='60s';
SET LOCAL lock_timeout='5s';
CREATE SCHEMA pireus_relocation_probe;
CREATE TABLE pireus_relocation_probe.docs(id integer PRIMARY KEY,body text,v public.vector(3));
INSERT INTO pireus_relocation_probe.docs VALUES(1,'pireus ontology','[1,2,3]'),(2,'unrelated sample','[1,2,4]');
CREATE INDEX probe_bm25 ON pireus_relocation_probe.docs USING bm25(id,body) WITH(key_field='id');
CREATE TABLE pireus_relocation_probe.base(id integer PRIMARY KEY);
SELECT pgivm.create_immv('pireus_relocation_probe.immv','SELECT count(*) AS n FROM pireus_relocation_probe.base');
INSERT INTO pireus_relocation_probe.base VALUES(1),(2);
DO $check$
BEGIN
 IF (SELECT count(*) FROM pireus_relocation_probe.docs WHERE body @@@ 'pireus') <> 1 THEN RAISE EXCEPTION 'BM25 mismatch'; END IF;
 IF ('[1,2,3]'::public.vector <-> '[1,2,4]'::public.vector) <> 1 THEN RAISE EXCEPTION 'vector mismatch'; END IF;
 IF public.ST_Distance(public.ST_Point(0,0),public.ST_Point(3,4)) <> 5 THEN RAISE EXCEPTION 'PostGIS mismatch'; END IF;
 IF (SELECT n FROM pireus_relocation_probe.immv) <> 2 THEN RAISE EXCEPTION 'IVM insert mismatch'; END IF;
END $check$;
DELETE FROM pireus_relocation_probe.base WHERE id=2;
DO $check$ BEGIN
 IF (SELECT n FROM pireus_relocation_probe.immv) <> 1 THEN RAISE EXCEPTION 'IVM delete mismatch'; END IF;
END $check$;
ROLLBACK;
SELECT 'RESTORED_DATABASE_FUNCTIONAL_PASS';

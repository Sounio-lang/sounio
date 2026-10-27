# Bounded CLDR metadata smoke input

current.bin is an unchanged copy of locale-meta-000000.bin from the
Slurm18677 complete CLDR49 metadata fixture generator. It contains the first
128 source-derived metadata rows and is the default input for the run-pass
unicode_locale_metadata.sio fixture.

SHA256: d4d7a7606dc080ced9ec29e8faa6532525e6fdb7274e806b14c3693397261a3c

Slurm18714 checked, compiled and natively executed this exact first chunk,
then separately executed and collected all72 chunks/9190 rows. This small
tracked fixture enables ordinary test discovery without an external corpus.
It must not be reported as full metadata or full collation conformance.

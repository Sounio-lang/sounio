# Exact-source CI wait

The remote tmux waiter polls run 34432396152, source a1cd764b9c, attempt 1, for at most two hours. Read-only observation errors remain observation errors and do not trigger resubmission. Non-success terminal CI or a changed run identity stops the waiter. Success invokes the pinned driver at most once; the driver independently checks exact-source checks and fresh pair/queue prerequisites. It does not retry a refused or failed driver. Terminal collection remains separate.

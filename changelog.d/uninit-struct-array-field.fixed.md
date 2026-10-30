An uninitialized struct now allocates each fixed array field before the first indexed store, instead of leaving that slot as handle 0.

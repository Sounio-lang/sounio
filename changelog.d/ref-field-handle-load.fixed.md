Loading or storing a field through `&T` or `&!T` no longer treats a bare handle id as an address. Implicit and explicit borrows both reach the field instead of faulting at that small address.

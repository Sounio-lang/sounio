# Chemical compiler crash: raw02 confirmation

The original two chemical sources and original deadlines were replayed with raw8abe, holding the archived whole-integrated01 source fixed. Job19583 captured both compiler139 failures and positive catalysis compile/native0. Job19586 adapted only the tracer address selector to the newly observed RIP0x5884ade, retained the same source and raw, and again captured both failures and the positive control.

Both frame captures report saved_report_site9036, current_fn=-1, current_func=-1. Disassembly at entry0x5884a04 confirms the Lowerer copy and reporter update followed by the49-word current-function name copy that dereferences-1. The diagnosis therefore also applies to this new raw. The cause of the invalid lowering state remains unresolved. No lowerer/source fix or production bypass is included.

Initial submission19581 used controller UID401 and failed launch with signal53 before any test executed; it is not a compiler result. The controller refused an attempted --uid1000 submission without assigning a job. Successful jobs19583/19586 used the worker submission route with UID1000. Their result directories and originals remain intact.

Archive19592 completed0; complete test sources, raw identities, C++ tracer inputs, native controls, outputs and final guards are preserved in custody/SHA256SUMS. This supplements integrated02 without editing its historical receipts.

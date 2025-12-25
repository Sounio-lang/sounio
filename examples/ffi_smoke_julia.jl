lib = get(ENV, "DEMETRIOS_LIB", "")
if isempty(lib)
    error("Set DEMETRIOS_LIB to the built shared library path (e.g., libffi_exports.so/.dylib/.dll)")
end

add = ccall((:add_i64, lib), Int64, (Int64, Int64), 2, 3)
@assert add == 5

out = fill(UInt8(0xAA), 8)
ccall((:write_u64_le, lib), Cvoid, (Ptr{UInt8}, UInt64), pointer(out), UInt64(4))
@assert out[1] == 0x04
@assert all(out[2:end] .== 0x00)

println("OK")


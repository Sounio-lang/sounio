open System
open Sounio.Interop

[<EntryPoint>]
let main argv =
    // Path to the souc binary — adjust for your system
    let soucPath =
        match argv |> Array.tryHead with
        | Some p -> p
        | None -> "bin/souc"

    printfn "=== Sounio <-> F# Interop Example ==="
    printfn ""

    // The SounioProcess wraps a souc subprocess running --serve.
    // It speaks the SNIO binary protocol over stdin/stdout.
    use sounio = new SounioProcess(soucPath)

    // 1. Health check
    printfn "[ping] %b" (sounio.Ping())

    // 2. Integer dot product
    let a = [| 1L; 2L; 3L; 4L; 5L |]
    let b = [| 10L; 20L; 30L; 40L; 50L |]
    let args = Array.append a b
    let dot = sounio.CallScalar("dot_product", args)
    printfn "[dot_product] %A . %A = %d" a b dot
    // Expected: 1*10 + 2*20 + 3*30 + 4*40 + 5*50 = 550

    // 3. Integer sum
    let vals = [| 100L; 200L; 300L |]
    let s = sounio.CallScalar("sum", vals)
    printfn "[sum] sum(%A) = %d" vals s
    // Expected: 600

    // 4. Integer vec_add
    let x = [| 1L; 2L; 3L |]
    let y = [| 10L; 20L; 30L |]
    let added = sounio.CallRaw("vec_add", Array.append x y)
    printfn "[vec_add] %A + %A = %A" x y added
    // Expected: [11; 22; 33]

    // 5. Integer vec_scale
    let scale_args = Array.append [| 5L |] [| 1L; 2L; 3L |]
    let scaled = sounio.CallRaw("vec_scale", scale_args)
    printfn "[vec_scale] 5 * [1;2;3] = %A" scaled
    // Expected: [5; 10; 15]

    // 6. Query server info (ABI, runtime version, limits)
    let info = sounio.Info()
    printfn "[info] abi=%d runtime=%d.%d.%d max_args=%d max_funcs=%d"
        info.[0] info.[1] info.[2] info.[3] info.[4] info.[5]

    // 7. Query capabilities bitmask
    let caps = sounio.Capabilities()
    let flags = [
        if caps &&& 1L <> 0L then "FFI"
        if caps &&& 2L <> 0L then "EXPORT"
        if caps &&& 4L <> 0L then "PROFILING"
        if caps &&& 8L <> 0L then "GC"
        if caps &&& 16L <> 0L then "SIMD"
        if caps &&& 32L <> 0L then "GPU"
        if caps &&& 64L <> 0L then "KERNEL"
    ]
    printfn "[capabilities] %d (%s)" caps (String.concat "+" flags)

    // 8. Health check via HEALTH message
    let healthy = sounio.Health()
    printfn "[health] ok=%b" healthy

    // 9. Initialize
    let inited = sounio.Init()
    printfn "[init] ack=%b" inited

    // 10. GPU capabilities
    let gpuCaps = sounio.GpuCaps()
    printfn "[gpu] caps=%d (GPU=%b)" gpuCaps (gpuCaps &&& 32L <> 0L)

    // 11. Session stats
    let msgCount = sounio.Stats()
    printfn "[stats] message_count=%d" msgCount

    printfn ""
    printfn "All interop calls completed successfully!"
    printfn "  Original: ping, dot_product, sum, vec_add, vec_scale"
    printfn "  New: info, capabilities, health, init, gpu_caps, stats"
    0

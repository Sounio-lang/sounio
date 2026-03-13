namespace Sounio.Interop

open System
open System.Diagnostics
open System.IO

/// Manages a Sounio subprocess running in --serve mode.
/// Provides typed function calls over the binary IPC protocol.
type SounioProcess(soucPath: string, ?programPath: string, ?extraArgs: string[]) =

    let mutable proc: Process option = None
    let mutable disposed = false

    let startProcess () =
        let psi = ProcessStartInfo()
        psi.FileName <- soucPath
        // If a program path is given: souc run <program> -- --serve
        // Otherwise: souc --serve (uses built-in server in compiler main)
        match programPath with
        | Some p ->
            psi.Arguments <- sprintf "run %s -- --serve" p
        | None ->
            psi.Arguments <- "--serve"
        match extraArgs with
        | Some args ->
            for a in args do
                psi.Arguments <- psi.Arguments + " " + a
        | None -> ()
        psi.UseShellExecute <- false
        psi.RedirectStandardInput <- true
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.CreateNoWindow <- true

        let p = Process.Start(psi)
        proc <- Some p

        // Read the ready signal (RESULT message with value 0)
        let resp = Protocol.readResponse p.StandardOutput.BaseStream
        match resp with
        | Protocol.Response.ResultValues [| 0L |] -> ()
        | Protocol.Response.ResultValues _ -> ()
        | Protocol.Response.ErrorMessage msg ->
            failwithf "Sounio process startup error: %s" msg
        | Protocol.Response.Shutdown ->
            failwith "Sounio process shut down during startup"

        p

    let ensureStarted () =
        match proc with
        | Some p when not p.HasExited -> p
        | _ -> startProcess ()

    /// Call a Sounio function with i64 arguments, returning i64 results.
    member _.CallRaw(funcName: string, args: int64[]) : int64[] =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeCallFunc stdin funcName args
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues values -> values
        | Protocol.Response.ErrorMessage msg ->
            failwithf "Sounio error in '%s': %s" funcName msg
        | Protocol.Response.Shutdown ->
            failwith "Unexpected shutdown"

    /// Call a Sounio function with f64 arguments, returning f64 results.
    member this.CallF64(funcName: string, args: float[]) : float[] =
        let i64Args = args |> Array.map BitConverter.DoubleToInt64Bits
        let results = this.CallRaw(funcName, i64Args)
        results |> Array.map BitConverter.Int64BitsToDouble

    /// Call a Sounio function with i64 arguments, returning a single i64.
    member this.CallScalar(funcName: string, args: int64[]) : int64 =
        let results = this.CallRaw(funcName, args)
        if results.Length = 0 then
            failwithf "Expected result from '%s', got empty" funcName
        results.[0]

    /// Call a Sounio function with f64 arguments, returning a single f64.
    member this.CallScalarF64(funcName: string, args: float[]) : float =
        let results = this.CallF64(funcName, args)
        if results.Length = 0 then
            failwithf "Expected result from '%s', got empty" funcName
        results.[0]

    /// Compute dot product of two f64 vectors.
    member this.DotProduct(a: float[], b: float[]) : float =
        if a.Length <> b.Length then
            invalidArg "b" "Vectors must have equal length"
        let args = Array.append a b
        this.CallScalarF64("dot_product", args)

    /// Element-wise vector addition.
    member this.VecAdd(a: float[], b: float[]) : float[] =
        if a.Length <> b.Length then
            invalidArg "b" "Vectors must have equal length"
        let args = Array.append a b
        this.CallF64("vec_add", args)

    /// Scalar-vector multiplication.
    member this.VecScale(scalar: float, v: float[]) : float[] =
        let args = Array.append [| scalar |] v
        this.CallF64("vec_scale", args)

    /// Sum all elements of a vector.
    member this.Sum(v: float[]) : float =
        this.CallScalarF64("sum", v)

    // ================================================================
    // KERNEL / SESSION EMBEDDING API
    // ================================================================

    /// Create a new kernel session.  Returns the session ID.
    member _.SessionCreate(?flags: int64) : int64 =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeSessionCreate stdin (defaultArg flags 0L)
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues [| sid |] -> sid
        | Protocol.Response.ErrorMessage msg ->
            failwithf "SessionCreate error: %s" msg
        | _ -> failwith "Unexpected response from SessionCreate"

    /// Destroy a kernel session.
    member _.SessionDestroy(sessionId: int64) =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeSessionDestroy stdin sessionId
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues [| 1L |] -> ()
        | Protocol.Response.ErrorMessage msg ->
            failwithf "SessionDestroy error: %s" msg
        | _ -> failwith "Unexpected response from SessionDestroy"

    /// Describe (compile) a kernel from a .sio source file.
    /// Returns (ok, kernel_id, param_count, return_count, strategy).
    member _.KernelDescribe(sessionId: int64, sourcePath: string, ?flags: int64) : int64[] =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeKernelDescribe stdin sessionId sourcePath (defaultArg flags 0L)
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues values -> values
        | Protocol.Response.ErrorMessage msg ->
            failwithf "KernelDescribe error: %s" msg
        | _ -> failwith "Unexpected response from KernelDescribe"

    /// Execute a described kernel with i64 arguments.
    member _.KernelExecute(sessionId: int64, kernelId: int64, args: int64[]) : int64[] =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeKernelExecute stdin sessionId kernelId args
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues values -> values
        | Protocol.Response.ErrorMessage msg ->
            failwithf "KernelExecute error: %s" msg
        | _ -> failwith "Unexpected response from KernelExecute"

    /// Retrieve session output (state, execute_count, error_count).
    member _.KernelOutput(sessionId: int64) : int64[] =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeKernelOutput stdin sessionId
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues values -> values
        | Protocol.Response.ErrorMessage msg ->
            failwithf "KernelOutput error: %s" msg
        | _ -> failwith "Unexpected response from KernelOutput"

    /// Retrieve structured diagnostics.
    member _.KernelDiagnostics(sessionId: int64) : int64[] =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeKernelDiagnostics stdin sessionId
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues values -> values
        | Protocol.Response.ErrorMessage msg ->
            failwithf "KernelDiagnostics error: %s" msg
        | _ -> failwith "Unexpected response from KernelDiagnostics"

    /// Retrieve artifact metadata.
    member _.KernelArtifacts(sessionId: int64) : int64[] =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeKernelArtifacts stdin sessionId
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues values -> values
        | Protocol.Response.ErrorMessage msg ->
            failwithf "KernelArtifacts error: %s" msg
        | _ -> failwith "Unexpected response from KernelArtifacts"


    /// Query server info: returns [abi_version, rt_major, rt_minor, rt_patch, max_args, max_funcs].
    member _.Info() : int64[] =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeInfo stdin
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues values -> values
        | Protocol.Response.ErrorMessage msg ->
            failwithf "Info error: %s" msg
        | _ -> failwith "Unexpected response from Info"

    /// Query capabilities bitmask: FFI=1, EXPORT=2, PROFILING=4, GC=8, SIMD=16, GPU=32, KERNEL=64.
    member _.Capabilities() : int64 =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeCapabilities stdin
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues [| v |] -> v
        | Protocol.Response.ErrorMessage msg ->
            failwithf "Capabilities error: %s" msg
        | _ -> failwith "Unexpected response from Capabilities"

    /// Health check via HEALTH message (not ping). Returns true if server responds 1.
    member _.Health() : bool =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeHealth stdin
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues [| 1L |] -> true
        | _ -> false

    /// Initialize session with config. Returns true if ack received.
    member _.Init() : bool =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeInit stdin
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues [| 1L |] -> true
        | _ -> false

    /// Query GPU capabilities.
    member _.GpuCaps() : int64 =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeGpuCaps stdin
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues [| v |] -> v
        | Protocol.Response.ErrorMessage msg ->
            failwithf "GpuCaps error: %s" msg
        | _ -> failwith "Unexpected response from GpuCaps"

    /// Query session statistics (message count).
    member _.Stats() : int64 =
        let p = ensureStarted ()
        let stdin = p.StandardInput.BaseStream
        let stdout = p.StandardOutput.BaseStream
        Protocol.writeStats stdin
        match Protocol.readResponse stdout with
        | Protocol.Response.ResultValues [| v |] -> v
        | Protocol.Response.ErrorMessage msg ->
            failwithf "Stats error: %s" msg
        | _ -> failwith "Unexpected response from Stats"

    /// Health check — returns true if the process responds.
    member this.Ping() : bool =
        try
            let result = this.CallScalar("ping", [||])
            result = 1L
        with _ -> false

    /// Gracefully shut down the Sounio process.
    member _.Shutdown() =
        match proc with
        | Some p when not p.HasExited ->
            try
                Protocol.writeShutdown p.StandardInput.BaseStream
                p.WaitForExit(5000) |> ignore
                if not p.HasExited then p.Kill()
            with _ ->
                try p.Kill() with _ -> ()
            proc <- None
        | _ ->
            proc <- None

    interface IDisposable with
        member this.Dispose() =
            if not disposed then
                disposed <- true
                this.Shutdown()

    /// Create a SounioProcess using the default souc binary path.
    static member CreateDefault(?programPath: string) =
        let defaultPath = "artifacts/omega/souc-bin/souc-linux-x86_64-jit"
        match programPath with
        | Some p -> new SounioProcess(defaultPath, p)
        | None -> new SounioProcess(defaultPath)

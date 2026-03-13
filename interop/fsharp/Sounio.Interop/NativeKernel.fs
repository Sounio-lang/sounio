namespace Sounio.Interop

open System
open System.Runtime.InteropServices

/// Loads a Sounio-compiled shared library (.so) and invokes exported functions
/// via .NET NativeLibrary API. This is the direct FFI path — no IPC overhead.
///
/// Usage:
///   use kernel = new NativeKernel("output.so")
///   let result = kernel.CallI64("add_i64", [| 3L; 4L |])
///   // result = 7L
type NativeKernel(libraryPath: string) =

    let handle =
        let h = NativeLibrary.Load(libraryPath)
        if h = IntPtr.Zero then
            failwithf "NativeKernel: failed to load '%s'" libraryPath
        h

    let mutable disposed = false

    /// Look up a function pointer by name. Throws if not found.
    member _.GetExport(funcName: string) : IntPtr =
        let addr = NativeLibrary.GetExport(handle, funcName)
        if addr = IntPtr.Zero then
            failwithf "NativeKernel: symbol '%s' not found in '%s'" funcName libraryPath
        addr

    /// Check if a symbol exists in the library.
    member _.HasExport(funcName: string) : bool =
        let mutable addr = IntPtr.Zero
        NativeLibrary.TryGetExport(handle, funcName, &addr)

    // ================================================================
    // Typed call helpers — SysV AMD64 ABI: args in rdi,rsi,rdx,rcx,r8,r9
    // All Sounio exported functions use i64 args and i64 return.
    // ================================================================

    /// Call an exported function with 0 i64 args, returning i64.
    member _.Call0(funcName: string) : int64 =
        let fp = NativeLibrary.GetExport(handle, funcName)
        let del = Marshal.GetDelegateForFunctionPointer<Func<int64>>(fp)
        del.Invoke()

    /// Call an exported function with 1 i64 arg, returning i64.
    member _.Call1(funcName: string, a: int64) : int64 =
        let fp = NativeLibrary.GetExport(handle, funcName)
        let del = Marshal.GetDelegateForFunctionPointer<Func<int64, int64>>(fp)
        del.Invoke(a)

    /// Call an exported function with 2 i64 args, returning i64.
    member _.Call2(funcName: string, a: int64, b: int64) : int64 =
        let fp = NativeLibrary.GetExport(handle, funcName)
        let del = Marshal.GetDelegateForFunctionPointer<Func<int64, int64, int64>>(fp)
        del.Invoke(a, b)

    /// Call an exported function with 3 i64 args, returning i64.
    member _.Call3(funcName: string, a: int64, b: int64, c: int64) : int64 =
        let fp = NativeLibrary.GetExport(handle, funcName)
        let del = Marshal.GetDelegateForFunctionPointer<Func<int64, int64, int64, int64>>(fp)
        del.Invoke(a, b, c)

    /// Call an exported function with 4 i64 args, returning i64.
    member _.Call4(funcName: string, a: int64, b: int64, c: int64, d: int64) : int64 =
        let fp = NativeLibrary.GetExport(handle, funcName)
        let del = Marshal.GetDelegateForFunctionPointer<Func<int64, int64, int64, int64, int64>>(fp)
        del.Invoke(a, b, c, d)

    /// Call an exported function with an i64 array (up to 6 args), returning i64.
    /// Dispatches to Call0..Call4 based on array length.
    member this.CallI64(funcName: string, args: int64[]) : int64 =
        match args.Length with
        | 0 -> this.Call0(funcName)
        | 1 -> this.Call1(funcName, args.[0])
        | 2 -> this.Call2(funcName, args.[0], args.[1])
        | 3 -> this.Call3(funcName, args.[0], args.[1], args.[2])
        | 4 -> this.Call4(funcName, args.[0], args.[1], args.[2], args.[3])
        | _ -> failwithf "NativeKernel.CallI64: too many args (%d), max 4 supported" args.Length

    /// Call with f64 args (bit-cast to i64), returning f64.
    member this.CallF64(funcName: string, args: float[]) : float =
        let i64Args = args |> Array.map BitConverter.DoubleToInt64Bits
        let result = this.CallI64(funcName, i64Args)
        BitConverter.Int64BitsToDouble(result)

    /// The path this kernel was loaded from.
    member _.LibraryPath = libraryPath

    /// The native library handle.
    member _.Handle = handle

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                NativeLibrary.Free(handle)

    /// Load a NativeKernel from the default output path.
    static member LoadDefault() =
        new NativeKernel("output.so")

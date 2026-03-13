namespace Sounio.Interop

open System
open System.IO

/// Binary wire protocol matching self-hosted/interop/protocol.sio.
/// All multi-byte values are little-endian.
[<RequireQualifiedAccess>]
module Protocol =

    /// Magic bytes: "SNIO" (0x53 0x4E 0x49 0x4F)
    let Magic = [| 0x53uy; 0x4Euy; 0x49uy; 0x4Fuy |]

    [<Literal>]
    let MsgCallFunc = 1uy

    [<Literal>]
    let MsgResult = 2uy

    [<Literal>]
    let MsgError = 3uy

    [<Literal>]
    let MsgShutdown = 4uy

    // Lifecycle message types (matches protocol.sio)
    [<Literal>]
    let MsgInfo = 5uy

    [<Literal>]
    let MsgCapabilities = 6uy

    [<Literal>]
    let MsgCompile = 7uy

    [<Literal>]
    let MsgDiagnostics = 9uy

    [<Literal>]
    let MsgHealth = 10uy

    [<Literal>]
    let MsgGpuCaps = 11uy

    [<Literal>]
    let MsgInit = 12uy

    [<Literal>]
    let MsgStats = 13uy

    // Kernel/session embedding ABI (matches protocol.sio msg types 14-20)
    [<Literal>]
    let MsgSessionCreate = 14uy

    [<Literal>]
    let MsgSessionDestroy = 15uy

    [<Literal>]
    let MsgKernelDescribe = 16uy

    [<Literal>]
    let MsgKernelExecute = 17uy

    [<Literal>]
    let MsgKernelOutput = 18uy

    [<Literal>]
    let MsgKernelDiagnostics = 19uy

    [<Literal>]
    let MsgKernelArtifacts = 20uy

    // ---- Writing ----

    let writeMagic (s: Stream) =
        s.Write(Magic, 0, 4)

    let writeU8 (s: Stream) (v: byte) =
        s.WriteByte(v)

    let writeU16LE (s: Stream) (v: uint16) =
        let buf = BitConverter.GetBytes(v)
        s.Write(buf, 0, 2)

    let writeU32LE (s: Stream) (v: uint32) =
        let buf = BitConverter.GetBytes(v)
        s.Write(buf, 0, 4)

    let writeI64LE (s: Stream) (v: int64) =
        let buf = BitConverter.GetBytes(v)
        s.Write(buf, 0, 8)

    /// Write a CALL_FUNC message: function name + array of i64 args (raw f64 bit patterns).
    let writeCallFunc (s: Stream) (funcName: string) (args: int64[]) =
        let nameBytes = Text.Encoding.UTF8.GetBytes(funcName)
        let nameLen = nameBytes.Length
        let argCount = int64 args.Length
        // body = 2 (name_len) + nameLen + 8 (arg_count) + args.Length * 8
        let bodyLen = uint32 (2 + nameLen + 8 + args.Length * 8)

        writeMagic s
        writeU8 s MsgCallFunc
        writeU32LE s bodyLen
        writeU16LE s (uint16 nameLen)
        s.Write(nameBytes, 0, nameLen)
        writeI64LE s argCount
        for a in args do
            writeI64LE s a
        s.Flush()

    /// Write a CALL_FUNC message with f64 arguments (auto bit-cast to i64).
    let writeCallFuncF64 (s: Stream) (funcName: string) (args: float[]) =
        let i64Args = args |> Array.map BitConverter.DoubleToInt64Bits
        writeCallFunc s funcName i64Args

    /// Write a SHUTDOWN message.
    let writeShutdown (s: Stream) =
        writeMagic s
        writeU8 s MsgShutdown
        writeU32LE s 0u
        s.Flush()


    /// Write an INFO message (empty payload).
    let writeInfo (s: Stream) =
        writeMagic s
        writeU8 s MsgInfo
        writeU32LE s 0u
        s.Flush()

    /// Write a CAPABILITIES message (empty payload).
    let writeCapabilities (s: Stream) =
        writeMagic s
        writeU8 s MsgCapabilities
        writeU32LE s 0u
        s.Flush()

    /// Write a HEALTH message (empty payload).
    let writeHealth (s: Stream) =
        writeMagic s
        writeU8 s MsgHealth
        writeU32LE s 0u
        s.Flush()

    /// Write an INIT message (empty payload).
    let writeInit (s: Stream) =
        writeMagic s
        writeU8 s MsgInit
        writeU32LE s 0u
        s.Flush()

    /// Write a GPU_CAPS message (empty payload).
    let writeGpuCaps (s: Stream) =
        writeMagic s
        writeU8 s MsgGpuCaps
        writeU32LE s 0u
        s.Flush()

    /// Write a STATS message (empty payload).
    let writeStats (s: Stream) =
        writeMagic s
        writeU8 s MsgStats
        writeU32LE s 0u
        s.Flush()

    /// Write a DIAGNOSTICS message (empty payload).
    let writeDiagnostics (s: Stream) =
        writeMagic s
        writeU8 s MsgDiagnostics
        writeU32LE s 0u
        s.Flush()

    /// Write a COMPILE message (empty payload — placeholder).
    let writeCompile (s: Stream) =
        writeMagic s
        writeU8 s MsgCompile
        writeU32LE s 0u
        s.Flush()

    /// Write a SESSION_CREATE message.
    /// Payload: [session_id_placeholder: i64 = 0][flags: i64]
    let writeSessionCreate (s: Stream) (flags: int64) =
        writeMagic s
        writeU8 s MsgSessionCreate
        writeU32LE s 16u  // 2 * 8 bytes
        writeI64LE s 0L   // placeholder (server assigns session_id)
        writeI64LE s flags
        s.Flush()

    /// Write a SESSION_DESTROY message.
    /// Payload: [session_id: i64]
    let writeSessionDestroy (s: Stream) (sessionId: int64) =
        writeMagic s
        writeU8 s MsgSessionDestroy
        writeU32LE s 8u
        writeI64LE s sessionId
        s.Flush()

    /// Write a KERNEL_DESCRIBE message.
    /// Payload: [session_id: i64][path_len: u16][path: bytes][flags: i64]
    let writeKernelDescribe (s: Stream) (sessionId: int64) (sourcePath: string) (flags: int64) =
        let pathBytes = Text.Encoding.UTF8.GetBytes(sourcePath)
        let pathLen = pathBytes.Length
        let bodyLen = uint32 (8 + 2 + pathLen + 8)
        writeMagic s
        writeU8 s MsgKernelDescribe
        writeU32LE s bodyLen
        writeI64LE s sessionId
        writeU16LE s (uint16 pathLen)
        s.Write(pathBytes, 0, pathLen)
        writeI64LE s flags
        s.Flush()

    /// Write a KERNEL_EXECUTE message.
    /// Payload: [session_id: i64][kernel_id: i64][arg_count: i64][args: i64[]]
    let writeKernelExecute (s: Stream) (sessionId: int64) (kernelId: int64) (args: int64[]) =
        let argCount = int64 args.Length
        let bodyLen = uint32 (8 + 8 + 8 + args.Length * 8)
        writeMagic s
        writeU8 s MsgKernelExecute
        writeU32LE s bodyLen
        writeI64LE s sessionId
        writeI64LE s kernelId
        writeI64LE s argCount
        for a in args do
            writeI64LE s a
        s.Flush()

    /// Write a KERNEL_OUTPUT message.
    /// Payload: [session_id: i64]
    let writeKernelOutput (s: Stream) (sessionId: int64) =
        writeMagic s
        writeU8 s MsgKernelOutput
        writeU32LE s 8u
        writeI64LE s sessionId
        s.Flush()

    /// Write a KERNEL_DIAGNOSTICS message.
    /// Payload: [session_id: i64]
    let writeKernelDiagnostics (s: Stream) (sessionId: int64) =
        writeMagic s
        writeU8 s MsgKernelDiagnostics
        writeU32LE s 8u
        writeI64LE s sessionId
        s.Flush()

    /// Write a KERNEL_ARTIFACTS message.
    /// Payload: [session_id: i64]
    let writeKernelArtifacts (s: Stream) (sessionId: int64) =
        writeMagic s
        writeU8 s MsgKernelArtifacts
        writeU32LE s 8u
        writeI64LE s sessionId
        s.Flush()

    // ---- Reading ----

    let private readExact (s: Stream) (buf: byte[]) (offset: int) (count: int) =
        let mutable pos = 0
        while pos < count do
            let n = s.Read(buf, offset + pos, count - pos)
            if n = 0 then
                raise (EndOfStreamException("Unexpected EOF reading from Sounio process"))
            pos <- pos + n

    let private readBytes (s: Stream) (count: int) =
        let buf = Array.zeroCreate count
        readExact s buf 0 count
        buf

    let private readU8 (s: Stream) =
        let b = s.ReadByte()
        if b = -1 then raise (EndOfStreamException("EOF"))
        byte b

    let private readU16LE (s: Stream) =
        let buf = readBytes s 2
        BitConverter.ToUInt16(buf, 0)

    let private readU32LE (s: Stream) =
        let buf = readBytes s 4
        BitConverter.ToUInt32(buf, 0)

    let private readI64LE (s: Stream) =
        let buf = readBytes s 8
        BitConverter.ToInt64(buf, 0)

    /// Parsed response from the Sounio process.
    [<Struct>]
    type Response =
        | ResultValues of values: int64[]
        | ErrorMessage of message: string
        | Shutdown

    /// Read one response message from the stream.
    let readResponse (s: Stream) : Response =
        // Validate magic
        let magic = readBytes s 4
        if magic <> Magic then
            failwithf "Invalid magic: %A" magic

        let msgType = readU8 s
        let bodyLen = readU32LE s

        match msgType with
        | x when x = MsgResult ->
            let count = readI64LE s
            let values = Array.init (int count) (fun _ -> readI64LE s)
            ResultValues values

        | x when x = MsgError ->
            let errLen = readU16LE s
            let errBytes = readBytes s (int errLen)
            ErrorMessage(Text.Encoding.UTF8.GetString(errBytes))

        | x when x = MsgShutdown ->
            Shutdown

        | _ ->
            // Skip unknown body
            let _ = readBytes s (int bodyLen)
            failwithf "Unknown message type: %d" msgType

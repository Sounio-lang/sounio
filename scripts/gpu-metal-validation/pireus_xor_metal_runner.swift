import Foundation
import Metal

guard CommandLine.arguments.count == 3 else {
    fputs("usage: pireus_xor_metal_runner <metallib> <kernel-name>\n", stderr)
    exit(2)
}
let kernelName = CommandLine.arguments[2]
guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("FAIL: no Metal device\n", stderr)
    exit(1)
}
guard let library = try? device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1])),
      let function = library.makeFunction(name: kernelName) else {
    fputs("FAIL: cannot load \(kernelName)\n", stderr)
    exit(1)
}

let pipeline = try! device.makeComputePipelineState(function: function)
let queue = device.makeCommandQueue()!
let byteCount = 16 * MemoryLayout<SIMD2<Float>>.stride

func cdSign(_ a: Int, _ b: Int, _ bits: Int) -> Int {
    if bits == 0 { return 1 }
    let half = 1 << (bits - 1)
    let ah = a >= half
    let bh = b >= half
    let al = a & (half - 1)
    let bl = b & (half - 1)
    if !ah && !bh { return cdSign(al, bl, bits - 1) }
    if !ah && bh { return cdSign(bl, al, bits - 1) }
    if ah && !bh { return bl == 0 ? cdSign(al, 0, bits - 1) : -cdSign(al, bl, bits - 1) }
    return bl == 0 ? -cdSign(0, al, bits - 1) : cdSign(bl, al, bits - 1)
}

func dispatch(_ inputA: [SIMD2<Float>], _ inputB: [SIMD2<Float>]) -> [SIMD2<Float>] {
    var a = inputA
    var b = inputB
    var output = Array(repeating: SIMD2<Float>(0, 0), count: 16)
    let aBuffer = device.makeBuffer(bytes: &a, length: byteCount, options: .storageModeShared)!
    let bBuffer = device.makeBuffer(bytes: &b, length: byteCount, options: .storageModeShared)!
    let outBuffer = device.makeBuffer(bytes: &output, length: byteCount, options: .storageModeShared)!
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(outBuffer, offset: 0, index: 2)
    encoder.dispatchThreads(MTLSize(width: 16, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        fputs("FAIL: Metal command did not complete: \(String(describing: command.error))\n", stderr)
        exit(1)
    }
    let values = outBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: 16)
    return (0..<16).map { values[$0] }
}

for i in 0..<16 {
    for j in 0..<16 {
        var a = Array(repeating: SIMD2<Float>(0, 0), count: 16)
        var b = Array(repeating: SIMD2<Float>(0, 0), count: 16)
        a[i] = SIMD2<Float>(1, 0)
        b[j] = SIMD2<Float>(1, 0)
        let output = dispatch(a, b)
        let destination = i ^ j
        for lane in 0..<16 {
            let expected = lane == destination ? Float(cdSign(i, j, 4)) : 0
            if output[lane].x != expected || output[lane].y != 0 {
                fputs("FAIL: basis i=\(i) j=\(j) lane=\(lane)\n", stderr)
                exit(1)
            }
        }
    }
}

let a = (0..<16).map { SIMD2<Float>(Float($0 + 1) / 7, Float(($0 % 3) - 1) * 1.0e-7) }
let b = (0..<16).map { SIMD2<Float>(Float(17 - $0) / 11, Float(($0 % 5) - 2) * 2.0e-7) }
let output = dispatch(a, b)
for d in 0..<16 {
    var expected = 0.0
    for j in 0..<16 {
        let av = Double(a[d ^ j].x) + Double(a[d ^ j].y)
        let bv = Double(b[j].x) + Double(b[j].y)
        expected += Double(cdSign(d ^ j, j, 4)) * av * bv
    }
    let actual = Double(output[d].x) + Double(output[d].y)
    if abs(actual - expected) > 2.0e-5 {
        fputs("FAIL: twofold lane=\(d) actual=\(actual) expected=\(expected)\n", stderr)
        exit(1)
    }
}

print("PIREUS_APPLE_METAL_XOR_RUNTIME_PASS device=\(device.name) basis_pairs=256 twofold_lanes=16 storage=float2-hi-lo")

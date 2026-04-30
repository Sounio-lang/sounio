import Metal
import Foundation

func approx(_ value: Float, _ expected: Float, _ tol: Float = 0.001) -> Bool {
    return abs(value - expected) <= tol
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("FAIL: no Metal device\n", stderr)
    exit(1)
}

let metallibURL = URL(fileURLWithPath: CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "./algebra_kernels.metallib")
guard let lib = try? device.makeLibrary(URL: metallibURL) else {
    fputs("FAIL: cannot load metallib at \(metallibURL.path)\n", stderr)
    exit(1)
}
let queue = device.makeCommandQueue()!

func runKernel(
    name: String,
    state: [Float],
    outCount: Int,
    shadowCount: Int
) -> ([Float], [Float]) {
    guard let fn = lib.makeFunction(name: name) else {
        fputs("FAIL: function \(name) not found\n", stderr)
        exit(1)
    }
    let pso = try! device.makeComputePipelineState(function: fn)
    var n: Int32 = 64
    var stateCopy = state
    var out = Array(repeating: Float(0), count: outCount)
    var shadow = Array(repeating: Float(0), count: shadowCount)

    let bufN = device.makeBuffer(bytes: &n, length: 4, options: .storageModeShared)!
    let bufState = device.makeBuffer(bytes: &stateCopy, length: stateCopy.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let bufOut = device.makeBuffer(bytes: &out, length: out.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let bufShadow = device.makeBuffer(bytes: &shadow, length: shadow.count * MemoryLayout<Float>.stride, options: .storageModeShared)!

    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(bufN, offset: 0, index: 0)
    enc.setBuffer(bufState, offset: 0, index: 1)
    enc.setBuffer(bufOut, offset: 0, index: 2)
    enc.setBuffer(bufShadow, offset: 0, index: 3)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    let outPtr = bufOut.contents().bindMemory(to: Float.self, capacity: out.count)
    let shadowPtr = bufShadow.contents().bindMemory(to: Float.self, capacity: shadow.count)
    for i in 0..<out.count { out[i] = outPtr[i] }
    for i in 0..<shadow.count { shadow[i] = shadowPtr[i] }
    return (out, shadow)
}

var ossmState = Array(repeating: Float(0), count: 128)
for i in 0..<64 {
    let v = Float(i + 1)
    ossmState[i] = v
    ossmState[i + 64] = v * 0.5
}
let (ossmOut, ossmShadow) = runKernel(
    name: "ossm_oct_step_f64_checked",
    state: ossmState,
    outCount: 128,
    shadowCount: 128
)
let ossmOK =
    approx(ossmOut[0], 0.5625) &&
    approx(ossmOut[64], 1.5625) &&
    approx(ossmOut[10], 4.9375) &&
    approx(ossmOut[74], 11.5625) &&
    approx(ossmShadow[0], 2.7734375)

var sedState = Array(repeating: Float(0), count: 256)
for i in 0..<64 {
    let v = Float(i + 1)
    sedState[i] = v
    sedState[i + 64] = v * 0.5
    sedState[i + 128] = v * 0.25
    sedState[i + 192] = v * 0.125
}
let (sedOut, sedShadow) = runKernel(
    name: "sedenion_cd_step_f64_checked",
    state: sedState,
    outCount: 128,
    shadowCount: 64
)
let sedOK =
    approx(sedOut[0], 1.703125) &&
    approx(sedOut[64], 1.5625) &&
    approx(sedOut[10], 128.734375) &&
    approx(sedOut[74], 72.1875) &&
    approx(sedShadow[0], 5.357666)

if ossmOK {
    print("PASS: gpu_metal_ossm_oct_step_f64_checked")
} else {
    print("FAIL: gpu_metal_ossm_oct_step_f64_checked")
}

if sedOK {
    print("PASS: gpu_metal_sedenion_cd_step_f64_checked")
} else {
    print("FAIL: gpu_metal_sedenion_cd_step_f64_checked")
}

exit(ossmOK && sedOK ? 0 : 1)

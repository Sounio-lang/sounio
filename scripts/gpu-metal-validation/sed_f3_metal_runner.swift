import Metal
import Foundation

// Reference state: rng(20260421).standard_normal(16) — matches gpu_sedenion_f3.sio
let hRef: [Float] = [
    -0.20220848, -0.64723789,  0.51044658,  1.04495879,
     0.38502833,  0.93319933,  0.15020544,  0.33725160,
    -0.93665325, -0.05310354,  0.98720954,  0.42369366,
    -0.04063361, -1.05319389, -0.53310586, -1.04790495
]

// sign_table[256] from stdlib/gpu/sedenion_kernels.sio sed_kernel_init()
let signTable: [Int32] = [
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,
    1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,
    1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,
    1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,-1,
    1,1,-1,1,-1,-1,-1,1,1,-1,1,-1,1,-1,1,-1,
    1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,
    1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,
    1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,
    1,1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,
    1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,
    1,1,1,1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,
    1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,
    1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,
    1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1
]

// triples[504] from stdlib/gpu/sedenion_kernels.sio sed_kernel_init()
let triplesFlat: [Int32] = [
    1,10,4,1,10,5,1,10,6,1,10,7,1,11,4,1,11,5,1,11,6,1,11,7,
    1,12,2,1,12,3,1,12,6,1,12,7,1,13,2,1,13,3,1,13,6,1,13,7,
    1,14,2,1,14,3,1,14,4,1,14,5,1,15,2,1,15,3,1,15,4,1,15,5,
    2,9,4,2,9,5,2,9,6,2,9,7,2,11,4,2,11,5,2,11,6,2,11,7,
    2,12,1,2,12,3,2,12,5,2,12,7,2,13,1,2,13,3,2,13,4,2,13,6,
    2,14,1,2,14,3,2,14,5,2,14,7,2,15,1,2,15,3,2,15,4,2,15,6,
    3,9,4,3,9,5,3,9,6,3,9,7,3,10,4,3,10,5,3,10,6,3,10,7,
    3,12,1,3,12,2,3,12,5,3,12,6,3,13,1,3,13,2,3,13,4,3,13,7,
    3,14,1,3,14,2,3,14,4,3,14,7,3,15,1,3,15,2,3,15,5,3,15,6,
    4,9,2,4,9,3,4,9,6,4,9,7,4,10,1,4,10,3,4,10,5,4,10,7,
    4,11,1,4,11,2,4,11,5,4,11,6,4,13,2,4,13,3,4,13,6,4,13,7,
    4,14,1,4,14,3,4,14,5,4,14,7,4,15,1,4,15,2,4,15,5,4,15,6,
    5,9,2,5,9,3,5,9,6,5,9,7,5,10,1,5,10,3,5,10,4,5,10,6,
    5,11,1,5,11,2,5,11,4,5,11,7,5,12,2,5,12,3,5,12,6,5,12,7,
    5,14,1,5,14,2,5,14,4,5,14,7,5,15,1,5,15,3,5,15,4,5,15,6,
    6,9,2,6,9,3,6,9,4,6,9,5,6,10,1,6,10,3,6,10,5,6,10,7,
    6,11,1,6,11,2,6,11,4,6,11,7,6,12,1,6,12,3,6,12,5,6,12,7,
    6,13,1,6,13,2,6,13,4,6,13,7,6,15,2,6,15,3,6,15,4,6,15,5,
    7,9,2,7,9,3,7,9,4,7,9,5,7,10,1,7,10,3,7,10,4,7,10,6,
    7,11,1,7,11,2,7,11,5,7,11,6,7,12,1,7,12,2,7,12,5,7,12,6,
    7,13,1,7,13,3,7,13,4,7,13,6,7,14,2,7,14,3,7,14,4,7,14,5
]

guard triplesFlat.count == 504 else { fatalError("triples count \(triplesFlat.count) != 504") }
guard signTable.count == 256 else { fatalError("signTable count \(signTable.count) != 256") }

// ---- Metal setup ----
guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("FAIL: no Metal device\n", stderr); exit(1)
}

let metallibURL = URL(fileURLWithPath: CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "./sedenion_f3.metallib")
guard let lib = try? device.makeLibrary(URL: metallibURL) else {
    fputs("FAIL: cannot load metallib at \(metallibURL.path)\n", stderr); exit(1)
}
guard let fn = lib.makeFunction(name: "sed_f3_l1_mass_kernel") else {
    fputs("FAIL: function sed_f3_l1_mass_kernel not found\n", stderr); exit(1)
}
let pso = try! device.makeComputePipelineState(function: fn)
let queue = device.makeCommandQueue()!

// ---- Buffers ----
var n: Int32 = 1
let bufN      = device.makeBuffer(bytes: &n, length: 4, options: .storageModeShared)!
let bufStates = device.makeBuffer(bytes: hRef, length: 64, options: .storageModeShared)!
let bufST     = device.makeBuffer(bytes: signTable, length: 1024, options: .storageModeShared)!
let bufTr     = device.makeBuffer(bytes: triplesFlat, length: 2016, options: .storageModeShared)!
var nTr: Int32 = 168
let bufNTr    = device.makeBuffer(bytes: &nTr, length: 4, options: .storageModeShared)!
let bufOut    = device.makeBuffer(length: 4, options: .storageModeShared)!

// ---- Dispatch ----
let cmd = queue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)
enc.setBuffer(bufN,      offset: 0, index: 0)
enc.setBuffer(bufStates, offset: 0, index: 1)
enc.setBuffer(bufST,     offset: 0, index: 2)
enc.setBuffer(bufTr,     offset: 0, index: 3)
enc.setBuffer(bufNTr,    offset: 0, index: 4)
enc.setBuffer(bufOut,    offset: 0, index: 5)
enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

// ---- Read back ----
let result = bufOut.contents().load(as: Float.self)
print(String(format: "F3 = %.6f  (expected 85.47±0.01)", result))
if result > 85.46 && result < 85.49 {
    print("PASS: gpu_sedenion_f3 Metal")
    exit(0)
} else {
    print("FAIL: out of range (\(result))")
    exit(1)
}

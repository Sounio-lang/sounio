import MetalKit
import SwiftUI
import LoomDomain

#if os(macOS)
import AppKit
#else
import UIKit
#endif

@MainActor
struct MetalFieldView {
    let accent: Color
    let motionEnabled: Bool
    let highOpacity: Bool
}

#if os(macOS)
extension MetalFieldView: NSViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> MTKView {
        makeView(context: context)
    }

    func updateNSView(_ view: MTKView, context: Context) {
        update(view: view, renderer: context.coordinator.renderer)
    }
}
#else
extension MetalFieldView: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> MTKView {
        makeView(context: context)
    }

    func updateUIView(_ view: MTKView, context: Context) {
        update(view: view, renderer: context.coordinator.renderer)
    }
}
#endif

extension MetalFieldView {
    final class Coordinator {
        fileprivate var renderer: MetalFieldRenderer?
    }

    fileprivate func makeView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.035, green: 0.045, blue: 0.06, alpha: 1)
        view.framebufferOnly = true
        view.preferredFramesPerSecond = motionEnabled ? 60 : 12
        view.isPaused = false
        view.enableSetNeedsDisplay = false

        if let renderer = try? MetalFieldRenderer(view: view) {
            context.coordinator.renderer = renderer
            view.delegate = renderer
            update(view: view, renderer: renderer)
        }
        return view
    }

    fileprivate func update(view: MTKView, renderer: MetalFieldRenderer?) {
        view.preferredFramesPerSecond = motionEnabled ? 60 : 12
        renderer?.motionEnabled = motionEnabled
        renderer?.opacity = highOpacity ? 1 : 0.35
        renderer?.accent = accent.metalComponents
    }
}

private extension Color {
    var metalComponents: SIMD3<Float> {
        #if os(macOS)
        let native = NSColor(self).usingColorSpace(.deviceRGB) ?? .cyan
        return SIMD3(Float(native.redComponent), Float(native.greenComponent), Float(native.blueComponent))
        #else
        let native = UIColor(self)
        var red: CGFloat = 0.26
        var green: CGFloat = 0.86
        var blue: CGFloat = 0.94
        var alpha: CGFloat = 1
        native.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        return SIMD3(Float(red), Float(green), Float(blue))
        #endif
    }
}

@MainActor
fileprivate final class MetalFieldRenderer: NSObject, MTKViewDelegate {
    private struct Uniforms {
        var timing: SIMD4<Float>
        var accent: SIMD4<Float>
    }

    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let startedAt = CFAbsoluteTimeGetCurrent()

    var accent = SIMD3<Float>(0.26, 0.86, 0.94)
    var opacity: Float = 1
    var motionEnabled = true

    init(view: MTKView) throws {
        guard let device = view.device,
              let queue = device.makeCommandQueue(),
              let url = Bundle.module.url(forResource: "LoomField", withExtension: "metal"),
              let source = try? String(contentsOf: url, encoding: .utf8)
        else {
            throw RendererError.missingMetal
        }

        let library = try device.makeLibrary(source: source, options: nil)
        guard let vertex = library.makeFunction(name: "loom_field_vertex"),
              let fragment = library.makeFunction(name: "loom_field_fragment")
        else {
            throw RendererError.missingMetal
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat

        commandQueue = queue
        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let buffer = commandQueue.makeCommandBuffer(),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: pass)
        else { return }

        let elapsed = motionEnabled ? Float(CFAbsoluteTimeGetCurrent() - startedAt) : 0
        var uniforms = Uniforms(
            timing: SIMD4(
                Float(view.drawableSize.width),
                Float(view.drawableSize.height),
                elapsed,
                motionEnabled ? 1 : 0
            ),
            accent: SIMD4(accent.x, accent.y, accent.z, opacity)
        )

        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()
        buffer.present(drawable)
        buffer.commit()
    }

    enum RendererError: Error {
        case missingMetal
    }
}

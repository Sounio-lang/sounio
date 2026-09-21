#if os(macOS)
import AppKit
import SwiftUI

@MainActor
struct WindowChrome: NSViewRepresentable {
    func makeNSView(context: Context) -> WindowProbeView {
        WindowProbeView(frame: .zero)
    }

    func updateNSView(_ view: WindowProbeView, context: Context) {
        view.configureWindow()
    }
}

@MainActor
final class WindowProbeView: NSView {
    private var didConfigure = false

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        configureWindow()
    }

    func configureWindow() {
        guard let window else { return }
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.sharingType = .readOnly

        guard !didConfigure else { return }
        didConfigure = true

        let visibleFrame = (window.screen ?? NSScreen.main)?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1380, height: 880)
        window.setContentSize(
            NSSize(
                width: min(1380, visibleFrame.width - 72),
                height: min(880, visibleFrame.height - 72)
            )
        )
        window.center()

        if let snapshotPath = Self.snapshotPath {
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(Self.snapshotDelaySeconds))
                self?.writeSnapshot(to: snapshotPath)
            }
        }
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        nil
    }

    private func writeSnapshot(to path: String) {
        guard let contentView = window?.contentView else { return }
        contentView.layoutSubtreeIfNeeded()
        let bounds = contentView.bounds
        guard let image = contentView.bitmapImageRepForCachingDisplay(in: bounds) else { return }
        contentView.cacheDisplay(in: bounds, to: image)
        guard let data = image.representation(using: .png, properties: [:]) else { return }
        try? data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    private static var snapshotPath: String? {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "--snapshot"),
              arguments.indices.contains(index + 1)
        else { return nil }
        return arguments[index + 1]
    }

    private static var snapshotDelaySeconds: Double {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "--snapshot-delay-seconds"),
              arguments.indices.contains(index + 1),
              let value = Double(arguments[index + 1])
        else { return 2 }
        return min(max(value, 0.5), 30)
    }
}
#endif

import SwiftUI

#if os(macOS)
import AppKit
#endif

@main
struct LoomSpatialApp: App {
    init() {
        #if os(macOS)
        NSApplication.shared.setActivationPolicy(.regular)
        #endif
    }

    var body: some Scene {
        #if os(macOS)
        WindowGroup("Loom Spatial") {
            LoomObservatoryView()
                .frame(minWidth: 760, minHeight: 640)
                .background(WindowChrome())
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1380, height: 880)
        #else
        WindowGroup("Loom Spatial") {
            LoomObservatoryView()
        }
        #endif
    }
}

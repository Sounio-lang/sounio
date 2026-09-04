import SwiftUI
import LoomDomain

struct LoomObservatoryView: View {
    @StateObject private var store = LoomStore()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                LoomColor.ink.ignoresSafeArea()
                MetalFieldView(
                    accent: store.dashboard.receipt.status.loomColor,
                    motionEnabled: !reduceMotion,
                    highOpacity: !reduceTransparency
                )
                .ignoresSafeArea()
                .accessibilityHidden(true)

                LinearGradient(
                    colors: [Color.black.opacity(0.18), .clear, Color.black.opacity(0.40)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                if proxy.size.width >= 1080 {
                    DesktopObservatory(store: store)
                } else {
                    CompactObservatory(store: store)
                }
            }
        }
        .preferredColorScheme(.dark)
        .task { await store.poll() }
    }
}

private struct DesktopObservatory: View {
    @ObservedObject var store: LoomStore
    @State private var mode: WorkbenchMode = .conversation

    var body: some View {
        VStack(spacing: 0) {
            ObservatoryToolbar(store: store, mode: $mode)
            workbench
                .padding(.horizontal, 12)
                .padding(.bottom, 12)
        }
    }

    @ViewBuilder
    private var workbench: some View {
        switch mode {
        case .conversation:
            HStack(spacing: 12) {
                LaneRail(store: store)
                    .frame(width: 236)
                ConversationSpaceView(store: store)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                ConversationContextDrawer(store: store)
                    .frame(width: 286)
            }
        case .spatial:
            HStack(spacing: 12) {
                LaneRail(store: store)
                    .frame(width: 226)
                VStack(spacing: 10) {
                    TopologyPanel(snapshot: store.dashboard, fleet: store.fleet, live: store.dashboardIsLive)
                    FabricSignals(snapshot: store.dashboard)
                }
                ConversationDock(store: store)
                    .frame(width: 380)
            }
        case .focus:
            ConversationSpaceView(store: store)
                .frame(maxWidth: 980, maxHeight: .infinity)
                .frame(maxWidth: .infinity)
        }
    }
}

private struct CompactObservatory: View {
    @ObservedObject var store: LoomStore
    @State private var selection = 0

    var body: some View {
        VStack(spacing: 0) {
            ObservatoryToolbar(store: store, mode: .constant(.conversation))
            Picker("Surface", selection: $selection) {
                Label("Weave", systemImage: "point.3.connected.trianglepath.dotted").tag(0)
                Label("Agents", systemImage: "bubble.left.and.bubble.right").tag(1)
                Label("Lanes", systemImage: "sidebar.left").tag(2)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 12)
            .padding(.bottom, 8)

            Group {
                switch selection {
                case 1: ConversationDock(store: store)
                case 2: LaneRail(store: store)
                default:
                    VStack(spacing: 8) {
                        TopologyPanel(snapshot: store.dashboard, fleet: store.fleet, live: store.dashboardIsLive)
                        FabricSignals(snapshot: store.dashboard)
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 10)
        }
    }
}

private struct ObservatoryToolbar: View {
    @ObservedObject var store: LoomStore
    @Binding var mode: WorkbenchMode

    var connectionColor: Color {
        store.connection == .connected ? LoomColor.green : LoomColor.amber
    }

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "point.3.filled.connected.trianglepath.dotted")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(LoomColor.cyan)
            VStack(alignment: .leading, spacing: 1) {
                Text("LOOM")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                Text("NATIVE WORKBENCH")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Divider().frame(height: 22)

            StatusPill(label: store.connection.label, color: connectionColor)
            StatusPill(
                label: store.dashboardIsLive
                    ? "Sounio \(store.dashboard.receipt.status.rawValue)"
                    : "scenario \(store.dashboard.receipt.status.rawValue)",
                color: store.dashboard.receipt.status.loomColor,
                systemImage: store.dashboardIsLive ? "checkmark.shield.fill" : "sparkles"
            )

            Spacer(minLength: 8)

            Picker("Workbench mode", selection: $mode) {
                ForEach(WorkbenchMode.allCases) { mode in
                    Label(mode.label, systemImage: mode.symbol).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 280)
            .accessibilityIdentifier("loom-workbench-mode")

            if store.connection == .connected {
                Label("Kernel decides", systemImage: "lock.shield")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(LoomColor.green)
                    .help("The backend owns routing decisions. Scenario previews are unavailable while live data is connected.")
                    .accessibilityIdentifier("loom-live-authority-indicator")
            } else {
                Menu {
                    Picker("Scenario", selection: $store.scenario) {
                        ForEach(DashboardScenario.allCases) { scenario in
                            Text(scenario.title).tag(scenario)
                        }
                    }
                } label: {
                    Label(store.scenario.title, systemImage: "switch.2")
                        .font(.system(size: 11, weight: .semibold))
                }
                .help("Preview an operational state while the live kernel is unavailable")
                .accessibilityIdentifier("loom-scenario-preview")
            }

            Button {
                Task { await store.refreshFleet() }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.plain)
            .help("Refresh Loom fleet")
        }
        .padding(.horizontal, 14)
        .frame(height: 54)
    }
}

private enum WorkbenchMode: String, CaseIterable, Identifiable {
    case conversation
    case spatial
    case focus

    var id: Self { self }

    var label: String {
        switch self {
        case .conversation: "Talk"
        case .spatial: "Fleet"
        case .focus: "Focus"
        }
    }

    var symbol: String {
        switch self {
        case .conversation: "bubble.left.and.bubble.right"
        case .spatial: "point.3.connected.trianglepath.dotted"
        case .focus: "rectangle.inset.filled"
        }
    }
}

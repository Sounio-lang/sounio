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

    var body: some View {
        VStack(spacing: 0) {
            ObservatoryToolbar(store: store)
            HStack(spacing: 10) {
                LaneRail(store: store)
                    .frame(width: 218)

                VStack(spacing: 10) {
                    TopologyPanel(snapshot: store.dashboard, fleet: store.fleet)
                    FabricSignals(snapshot: store.dashboard)
                }

                ConversationDock(store: store)
                    .frame(width: 342)
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 10)
        }
    }
}

private struct CompactObservatory: View {
    @ObservedObject var store: LoomStore
    @State private var selection = 0

    var body: some View {
        VStack(spacing: 0) {
            ObservatoryToolbar(store: store)
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
                        TopologyPanel(snapshot: store.dashboard, fleet: store.fleet)
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
                Text("SPATIAL OBSERVATORY")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Divider().frame(height: 22)

            StatusPill(label: store.connection.label, color: connectionColor)
            StatusPill(
                label: "scenario \(store.dashboard.receipt.status.rawValue)",
                color: store.dashboard.receipt.status.loomColor,
                systemImage: "sparkles"
            )

            Spacer(minLength: 8)

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
            .help("Preview operational state")

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

import SwiftUI
import LoomDomain

struct LaneRail: View {
    @ObservedObject var store: LoomStore

    private var lanes: [LoomFleetSnapshot.Lane] {
        (store.fleet?.lanes ?? []).sorted {
            let lhsRank = $0.displayRank
            let rhsRank = $1.displayRank
            if lhsRank != rhsRank { return lhsRank < rhsRank }
            if $0.agent != $1.agent { return $0.agent.localizedStandardCompare($1.agent) == .orderedAscending }
            return $0.lane.localizedStandardCompare($1.lane) == .orderedAscending
        }
    }

    var body: some View {
        GlassSurface {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("ACTIVE LANES")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(lanes.count))
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(LoomColor.cyan)
                }
                .padding(12)

                Divider().opacity(0.55)

                ScrollView {
                    LazyVStack(spacing: 2) {
                        if lanes.isEmpty {
                            MockLaneRow(
                                agent: "codex-3",
                                lane: "loom-material",
                                state: "live",
                                selected: true
                            )
                            MockLaneRow(
                                agent: "claude-2",
                                lane: "compiler-parity",
                                state: "claimed",
                                selected: false
                            )
                            MockLaneRow(
                                agent: "grok-cli2",
                                lane: "hostile-review",
                                state: "unresponsive",
                                selected: false
                            )
                        } else {
                            ForEach(lanes) { lane in
                                Button {
                                    store.selectedLaneId = lane.id
                                } label: {
                                    LaneRow(lane: lane, selected: store.selectedLaneId == lane.id)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .padding(6)
                }

                Divider().opacity(0.55)
                HStack(spacing: 8) {
                    Image(systemName: "lock.shield")
                        .foregroundStyle(LoomColor.green)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("KERNEL AUTHORITY")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                        Text("UI projection only")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .padding(12)
            }
        }
    }
}

private struct LaneRow: View {
    let lane: LoomFleetSnapshot.Lane
    let selected: Bool

    var body: some View {
        MockLaneRow(
            agent: lane.agent,
            lane: lane.lane,
            state: lane.displayState,
            selected: selected
        )
    }
}

private struct MockLaneRow: View {
    let agent: String
    let lane: String
    let state: String
    let selected: Bool

    private var color: Color {
        switch state {
        case "active", "live": LoomColor.green
        case "claimed", "recoverable": LoomColor.amber
        case "unresponsive", "orphaned", "lost": LoomColor.red
        default: .secondary
        }
    }

    var body: some View {
        HStack(spacing: 9) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
                .shadow(color: color.opacity(0.75), radius: 5)
            VStack(alignment: .leading, spacing: 2) {
                Text(agent)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.primary)
                Text(lane)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 4)
            if selected {
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(LoomColor.cyan)
            }
        }
        .padding(.horizontal, 9)
        .frame(height: 48)
        .background(
            selected ? LoomColor.cyan.opacity(0.10) : Color.clear,
            in: RoundedRectangle(cornerRadius: 6)
        )
        .overlay(alignment: .leading) {
            if selected {
                Capsule().fill(LoomColor.cyan).frame(width: 2, height: 25)
            }
        }
    }
}

struct TopologyPanel: View {
    let snapshot: DashboardSnapshot
    let fleet: LoomFleetSnapshot?
    let live: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 7) {
                        Text("ROUTE TOPOLOGY")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                        Text(live ? "LIVE AUTHORITY" : "SCENARIO")
                            .font(.system(size: 8, weight: .black, design: .monospaced))
                            .foregroundStyle(LoomColor.amber)
                    }
                    Text("The fabric is thinking.")
                        .font(.system(size: 23, weight: .medium, design: .rounded))
                    Text("Task -> RouteDecision -> RouteReceipt")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if let fleet {
                    StatusPill(
                        label: "\(fleet.summary.live)/\(fleet.summary.lanes) live",
                        color: fleet.coordinationAvailable ? LoomColor.green : LoomColor.amber
                    )
                }
                StatusPill(
                    label: live
                        ? "9032 \(snapshot.receipt.status.rawValue)"
                        : "simulated \(snapshot.receipt.status.rawValue)",
                    color: snapshot.receipt.status.loomColor
                )
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            TopologyWeaveView(snapshot: snapshot)
                .frame(minHeight: 350)
                .accessibilityLabel("Routing topology scenario")
                .accessibilityValue(snapshot.receipt.reason)

            RouteReceiptStrip(receipt: snapshot.receipt, live: live)
        }
        .background(Color.black.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(LoomColor.cyan.opacity(0.12), lineWidth: 0.7)
        )
    }
}

private struct RouteReceiptStrip: View {
    let receipt: RouteReceipt
    let live: Bool

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: receipt.status == .committed ? "checkmark.seal.fill" : "xmark.seal.fill")
                    .foregroundStyle(receipt.status.loomColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text(live ? "SOUNIO ROUTE RECEIPT · ACTION 9032" : "SCENARIO RECEIPT · NOT AUTHORITY")
                        .font(.system(size: 8, weight: .black, design: .monospaced))
                        .foregroundStyle(LoomColor.amber)
                    Text(receipt.reason)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(2)
                }
                Spacer(minLength: 8)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 14) {
                    ReceiptField(name: "taskId", value: receipt.taskId)
                    ReceiptField(name: "policy", value: receipt.policy)
                    ReceiptField(name: "poolId", value: receipt.poolId)
                    ReceiptField(name: "adapterId", value: receipt.adapterId)
                    ReceiptField(name: "model", value: receipt.model)
                    ReceiptField(name: "effort", value: receipt.effort)
                    ReceiptField(name: "fallbackChain", value: receipt.fallbackChain.joined(separator: " -> ").nilIfEmpty ?? "none")
                    ReceiptField(name: "status", value: receipt.status.rawValue)
                    if let semanticsHash = receipt.semanticsHash {
                        ReceiptField(name: "semanticsHash", value: String(semanticsHash.prefix(16)) + "...")
                    }
                    if let sessionId = receipt.sessionId, !sessionId.isEmpty {
                        ReceiptField(name: "sessionId", value: sessionId)
                    }
                }
            }
        }
        .padding(12)
    }
}

private struct ReceiptField: View {
    let name: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name)
                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .lineLimit(1)
        }
    }
}

struct FabricSignals: View {
    let snapshot: DashboardSnapshot

    var body: some View {
        GlassSurface {
            HStack(spacing: 18) {
                Signal(
                    title: "POOL",
                    value: snapshot.pools[0].health.rawValue,
                    color: snapshot.pools[0].health.loomColor
                )
                Signal(
                    title: "ADAPTER",
                    value: snapshot.adapters[0].health.rawValue,
                    color: snapshot.adapters[0].health.loomColor
                )
                Signal(
                    title: "QUOTA",
                    value: snapshot.pools[0].state.rawValue,
                    color: snapshot.pools[0].state == .unknown ? LoomColor.amber : LoomColor.cyan
                )
                Signal(
                    title: "OWNER",
                    value: snapshot.task.owner,
                    color: snapshot.scenario == .ownershipBlock ? LoomColor.red : LoomColor.green
                )
                Spacer(minLength: 0)
                Image(systemName: "waveform.path.ecg.rectangle")
                    .font(.system(size: 20, weight: .light))
                    .foregroundStyle(LoomColor.magenta)
            }
            .padding(.horizontal, 14)
            .frame(height: 54)
        }
    }
}

private struct Signal: View {
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(value.uppercased())
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(color)
                .lineLimit(1)
        }
    }
}

struct ConversationDock: View {
    @ObservedObject var store: LoomStore
    @State private var tab = 0

    private var selectedLane: LoomFleetSnapshot.Lane? { store.selectedLane }
    private var selectedLaneColor: Color { selectedLane?.displayColor ?? LoomColor.amber }
    private var messageStatusColor: Color {
        switch store.messageState {
        case .ready:
            selectedLane?.deliveryReadiness == .immediate ? LoomColor.green : LoomColor.amber
        case .accepted: LoomColor.green
        case .sending: LoomColor.cyan
        case .failed: LoomColor.red
        case .unconfigured: LoomColor.amber
        }
    }
    private var messageStatusIcon: String {
        switch store.messageState {
        case .ready:
            selectedLane?.deliveryReadiness == .immediate
                ? "antenna.radiowaves.left.and.right"
                : "archivebox.fill"
        case .sending: "arrow.trianglehead.2.clockwise.rotate.90"
        case .accepted: "checkmark.seal.fill"
        case .failed: "exclamationmark.octagon.fill"
        case .unconfigured: "cable.connector.slash"
        }
    }
    private var messageStatusLabel: String {
        guard case .ready = store.messageState else { return store.messageState.label }
        guard let selectedLane else { return "DURABLE BUS READY / SELECT A LANE" }
        if selectedLane.deliveryReadiness == .immediate {
            return "DURABLE BUS + ACTIVE ENDPOINT"
        }
        let endpoint = selectedLane.endpointState.isEmpty
            ? "UNKNOWN"
            : selectedLane.endpointState.uppercased()
        return "DURABLE ONLY / ENDPOINT \(endpoint)"
    }
    private var messageStatusDetail: String? {
        switch store.messageState {
        case let .accepted(receipt): receipt.messageId
        case let .failed(reason): reason
        default: store.selectedThread?.request.id
        }
    }

    var body: some View {
        GlassSurface {
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("AGENT CHANNEL")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                        Text(selectedLane.map { "\($0.agent) / \($0.lane)" } ?? "No live lane selected")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    Circle().fill(selectedLaneColor).frame(width: 7, height: 7)
                        .shadow(color: selectedLaneColor.opacity(0.8), radius: 5)
                }
                .padding(12)

                Picker("Channel", selection: $tab) {
                    Text("Conversation").tag(0)
                    Text("Evidence").tag(1)
                    Text("Configure").tag(2)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .padding(.horizontal, 10)
                .padding(.bottom, 10)

                Divider().opacity(0.55)

                if let selectedLane {
                    LaneContextStrip(lane: selectedLane)
                    Divider().opacity(0.35)
                }

                ScrollView {
                    if tab == 0 {
                        LazyVStack(spacing: 12) {
                            HStack {
                                Text("THREAD TRUTH")
                                    .font(.system(size: 8, weight: .black, design: .monospaced))
                                    .foregroundStyle(store.visibleThreadState == "answered" ? LoomColor.green : LoomColor.cyan)
                                Spacer()
                                Text(store.visibleThreadState?.replacingOccurrences(of: "_", with: " ").uppercased() ?? "NO DURABLE THREAD")
                                    .font(.system(size: 8, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                            if let error = store.threadError {
                                ThreadStateCard(
                                    title: "THREAD READ REFUSED",
                                    detail: error,
                                    color: LoomColor.red
                                )
                            } else if store.visibleThreadEvents.isEmpty {
                                ThreadStateCard(
                                    title: store.messageBridgeConfigured ? "NO THREAD FOR THIS LANE" : "MESSAGE BRIDGE NOT CONFIGURED",
                                    detail: selectedLane?.deliveryReadiness == .immediate
                                        ? "ACTIVE ENDPOINT / READY FOR A DURABLE REQUEST"
                                        : "DURABLE BUS AVAILABLE WHEN CONFIGURED",
                                    color: selectedLane?.deliveryReadiness == .immediate ? LoomColor.green : LoomColor.amber
                                )
                            } else {
                                ForEach(store.visibleThreadEvents) { event in
                                    ThreadEventBubble(event: event)
                                }
                            }
                        }
                        .padding(12)
                    } else if tab == 1 {
                        EvidenceLedger(snapshot: store.dashboard, eventGroups: store.eventGroups)
                            .padding(12)
                    } else {
                        RoutingConfigurationPanel(store: store)
                            .padding(12)
                    }
                }

                if tab == 0 {
                    Divider().opacity(0.55)

                    VStack(alignment: .leading, spacing: 7) {
                        HStack(alignment: .bottom, spacing: 8) {
                            TextField("Message selected agent", text: $store.conversationDraft, axis: .vertical)
                                .textFieldStyle(.plain)
                                .font(.system(size: 12))
                                .lineLimit(1...4)
                                .padding(9)
                                .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 6))
                                .accessibilityIdentifier("loom-conversation-draft")
                                .accessibilityLabel("Message selected agent")
                            Button {
                                Task { await store.sendMessage() }
                            } label: {
                                Image(systemName: "arrow.up.circle.fill")
                                    .font(.system(size: 24))
                                    .foregroundStyle(store.canSendMessage ? LoomColor.cyan : .secondary)
                            }
                            .buttonStyle(.plain)
                            .disabled(!store.canSendMessage)
                            .keyboardShortcut(.return, modifiers: [.command])
                            .help("Send through the authenticated Loom message bridge")
                            .accessibilityIdentifier("loom-conversation-send")
                            .accessibilityLabel("Send durable message")
                        }

                        HStack(spacing: 6) {
                            Image(systemName: messageStatusIcon)
                                .foregroundStyle(messageStatusColor)
                            Text(messageStatusLabel)
                                .foregroundStyle(messageStatusColor)
                            Spacer(minLength: 6)
                            if let messageStatusDetail {
                                Text(messageStatusDetail)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                        }
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                    }
                    .padding(10)
                }
            }
        }
    }
}

private struct RoutingConfigurationPanel: View {
    @ObservedObject var store: LoomStore

    private var policy: Binding<String> {
        Binding(get: { store.routingDraft.policy }, set: { store.setRoutingPolicy($0) })
    }

    private var model: Binding<String> {
        Binding(get: { store.routingDraft.model }, set: { store.setRoutingModel($0) })
    }

    private var effort: Binding<String> {
        Binding(get: { store.routingDraft.effort }, set: { store.setRoutingEffort($0) })
    }

    private var statusColor: Color {
        switch store.routingConfigState {
        case .ready, .stored: LoomColor.green
        case .editing: LoomColor.amber
        case .loading, .saving: LoomColor.cyan
        case .failed: LoomColor.red
        case .unconfigured: LoomColor.amber
        }
    }

    private var statusIcon: String {
        switch store.routingConfigState {
        case .ready: "checkmark.shield.fill"
        case .stored: "checkmark.seal.fill"
        case .editing: "pencil.and.list.clipboard"
        case .loading, .saving: "arrow.trianglehead.2.clockwise.rotate.90"
        case .failed: "exclamationmark.octagon.fill"
        case .unconfigured: "cable.connector.slash"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("ROUTING CONFIG")
                        .font(.system(size: 10, weight: .black, design: .monospaced))
                    Text("DECLARATIVE INPUT / BACKEND ARBITRATES")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    Task { await store.refreshRoutingConfig() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 14, weight: .semibold))
                }
                .buttonStyle(.plain)
                .help("Reload the backend routing configuration")
                .accessibilityLabel("Reload routing configuration")
            }

            HStack(spacing: 7) {
                Image(systemName: statusIcon).foregroundStyle(statusColor)
                Text(store.routingConfigState.label)
                    .foregroundStyle(statusColor)
                Spacer(minLength: 8)
                if let config = store.routingConfig {
                    Text("REV \(config.revision)")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.system(size: 8, weight: .bold, design: .monospaced))

            if case let .failed(reason) = store.routingConfigState {
                ThreadStateCard(title: "CONFIGURATION REFUSED", detail: reason, color: LoomColor.red)
            }

            VStack(alignment: .leading, spacing: 8) {
                Picker("Policy", selection: policy) {
                    Text("Authority first").tag("authority-first")
                    Text("Capacity aware").tag("capacity-aware")
                    Text("Latency aware").tag("latency-aware")
                }
                .pickerStyle(.menu)

                Picker("Model", selection: model) {
                    Text("Terra").tag("gpt-5.6-terra")
                    Text("Sol").tag("gpt-5.6-sol")
                }
                .pickerStyle(.menu)

                Picker("Effort", selection: effort) {
                    Text("Low").tag("low")
                    Text("Medium").tag("medium")
                    Text("High").tag("high")
                }
                .pickerStyle(.segmented)
            }
            .font(.system(size: 11, weight: .medium))

            VStack(alignment: .leading, spacing: 7) {
                configOrder("POOL ORDER", store.routingDraft.poolOrder, color: LoomColor.cyan)
                configOrder("ADAPTER ORDER", store.routingDraft.adapterOrder, color: LoomColor.magenta)
            }

            HStack(spacing: 10) {
                Button {
                    Task { await store.saveRoutingConfig() }
                } label: {
                    Image(systemName: "square.and.arrow.down.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(store.canSaveRoutingConfig ? LoomColor.cyan : .secondary)
                }
                .buttonStyle(.plain)
                .disabled(!store.canSaveRoutingConfig)
                .help("Store declarative routing configuration in the authenticated backend")
                .accessibilityIdentifier("loom-routing-save")
                .accessibilityLabel("Store routing configuration")

                VStack(alignment: .leading, spacing: 2) {
                    Text("CONFIGURATION IS NOT A DECISION")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                    Text("Routes and receipts remain backend-produced evidence.")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }

            if case let .stored(receipt) = store.routingConfigState {
                HStack(spacing: 6) {
                    Image(systemName: "number.circle.fill").foregroundStyle(LoomColor.green)
                    Text("RECEIPT REV \(receipt.revision)")
                    Text(String(receipt.digest.prefix(12)) + "...")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .font(.system(size: 8, weight: .bold, design: .monospaced))
            }

            Divider().opacity(0.45)

            VStack(alignment: .leading, spacing: 9) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("ROUTE A REVIEW")
                            .font(.system(size: 10, weight: .black, design: .monospaced))
                        Text("TASK -> SOUNIO 9032 -> PROVIDER CUSTODY")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    routeStateIndicator
                }

                TextField("Task title", text: $store.routeTitle)
                    .textFieldStyle(.plain)
                    .font(.system(size: 11, weight: .semibold))
                    .padding(8)
                    .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 6))
                    .accessibilityIdentifier("loom-route-title")

                TextField("Review brief", text: $store.routePrompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 11))
                    .lineLimit(3...7)
                    .padding(8)
                    .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 6))
                    .accessibilityIdentifier("loom-route-prompt")

                HStack(spacing: 9) {
                    Button {
                        Task { await store.routeTask() }
                    } label: {
                        Image(systemName: "point.3.connected.trianglepath.dotted")
                            .font(.system(size: 21, weight: .semibold))
                            .foregroundStyle(store.canRouteTask ? LoomColor.magenta : .secondary)
                    }
                    .buttonStyle(.plain)
                    .disabled(!store.canRouteTask)
                    .help("Submit a review task to the Sounio routing authority")
                    .accessibilityIdentifier("loom-route-submit")
                    .accessibilityLabel("Route review task")

                    VStack(alignment: .leading, spacing: 2) {
                        Text(store.routeState.label)
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundStyle(routeStateColor)
                        Text("External models remain REVIEW_ONLY.")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 0)
                }

                if case let .failed(reason) = store.routeState {
                    ThreadStateCard(title: "ROUTING FAILED CLOSED", detail: reason, color: LoomColor.red)
                }
            }
        }
    }

    private var routeStateColor: Color {
        switch store.routeState {
        case .ready: LoomColor.cyan
        case .deciding: LoomColor.magenta
        case let .received(operation): operation.receipt.status.loomColor
        case .failed: LoomColor.red
        }
    }

    @ViewBuilder
    private var routeStateIndicator: some View {
        switch store.routeState {
        case .deciding:
            ProgressView().controlSize(.small).tint(LoomColor.magenta)
        case let .received(operation):
            Image(systemName: operation.receipt.status == .running
                ? "checkmark.shield.fill" : "xmark.shield.fill")
                .foregroundStyle(operation.receipt.status.loomColor)
        case .failed:
            Image(systemName: "exclamationmark.octagon.fill").foregroundStyle(LoomColor.red)
        case .ready:
            Image(systemName: "shield.lefthalf.filled").foregroundStyle(LoomColor.cyan)
        }
    }

    private func configOrder(_ label: String, _ values: [String], color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)
            ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                HStack(spacing: 7) {
                    Text(String(format: "%02d", index + 1))
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(color)
                    Text(value)
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
                .padding(.vertical, 5)
                .padding(.horizontal, 7)
                .background(Color.black.opacity(0.14), in: RoundedRectangle(cornerRadius: 5))
            }
        }
    }
}

private struct LaneContextStrip: View {
    let lane: LoomFleetSnapshot.Lane

    var body: some View {
        HStack(spacing: 12) {
            contextField("STATE", lane.displayState.uppercased(), lane.displayColor)
            contextField(
                "ENDPOINT",
                lane.endpointState.isEmpty ? "UNKNOWN" : lane.endpointState.uppercased(),
                lane.hasActiveEndpoint ? LoomColor.green : LoomColor.amber
            )
            contextField("HARNESS", lane.harness.uppercased(), LoomColor.cyan)
            contextField("CURSOR", String(lane.cursor), LoomColor.magenta)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .frame(height: 42)
    }

    private func contextField(_ name: String, _ value: String, _ color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name)
                .font(.system(size: 7, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(color)
                .lineLimit(1)
        }
    }
}

private struct ThreadEventBubble: View {
    let event: LoomThreadEvent

    var body: some View {
        HStack {
            if event.isLocal { Spacer(minLength: 28) }
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    Text(event.kind.replacingOccurrences(of: "_", with: " ").uppercased())
                    Text(event.state.uppercased())
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                    Text(event.utc)
                        .foregroundStyle(.secondary)
                }
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundStyle(event.isLocal ? LoomColor.cyan : LoomColor.magenta)
                Text(event.actor.uppercased())
                    .font(.system(size: 8, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text(event.body)
                    .font(.system(size: 11))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(10)
            .background(
                (event.isLocal ? LoomColor.cyan : LoomColor.magenta).opacity(0.08),
                in: RoundedRectangle(cornerRadius: 7)
            )
            if !event.isLocal { Spacer(minLength: 28) }
        }
    }
}

private struct ThreadStateCard: View {
    let title: String
    let detail: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(color)
            Text(detail)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(color.opacity(0.26)))
    }
}

private struct ReceiptEvidence: View {
    let receipt: RouteReceipt

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: "doc.text.magnifyingglass")
                .foregroundStyle(receipt.status.loomColor)
            VStack(alignment: .leading, spacing: 4) {
                Text(receipt.producingLanguage == "Sounio"
                    ? "LIVE SOUNIO RECEIPT · ACTION 9032"
                    : "SCENARIO RECEIPT · NOT AUTHORITY")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text(receipt.taskId)
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                Text(receipt.status.rawValue.uppercased())
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(receipt.status.loomColor)
            }
            Spacer()
        }
        .padding(10)
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(receipt.status.loomColor.opacity(0.24)))
    }
}

private struct EvidenceLedger: View {
    let snapshot: DashboardSnapshot
    let eventGroups: [LoomEventGroup]

    private var verifiedCount: Int { eventGroups.filter(\.verified).count }
    private var eventCount: Int { eventGroups.reduce(0) { $0 + $1.events.count } }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("\(verifiedCount)/\(eventGroups.count) journals verified", systemImage: "checkmark.seal")
                .foregroundStyle(eventGroups.isEmpty ? LoomColor.amber : LoomColor.green)
            Label("\(eventCount) observed events", systemImage: "waveform.path.ecg")
            Label("Decision is backend-owned", systemImage: "server.rack")
            Label("Projection is read-only", systemImage: "eye")
            Label("Receipt is not semantic authority", systemImage: "checkmark.shield")
            Divider().opacity(0.55)
            Text(snapshot.receipt.reason)
                .foregroundStyle(.secondary)
        }
        .font(.system(size: 11, weight: .medium))
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}

private extension LoomFleetSnapshot.Lane {
    var displayState: String {
        if !presenceState.isEmpty && presenceState != "none" { return presenceState }
        return state
    }

    var displayRank: Int {
        switch displayState {
        case "live", "active": 0
        case "claimed", "recoverable": 1
        case "unresponsive": 2
        case "orphaned", "lost": 3
        default: 4
        }
    }

    var displayColor: Color {
        switch displayState {
        case "live", "active": LoomColor.green
        case "claimed", "recoverable": LoomColor.amber
        case "unresponsive", "orphaned", "lost": LoomColor.red
        default: .secondary
        }
    }
}

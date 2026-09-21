import SwiftUI
import LoomDomain

struct ConversationSpaceView: View {
    @ObservedObject var store: LoomStore

    private var selectedLane: LoomFleetSnapshot.Lane? { store.selectedLane }
    private var accent: Color { selectedLane?.displayColor ?? LoomColor.cyan }

    var body: some View {
        GlassSurface(radius: 16) {
            VStack(spacing: 0) {
                conversationHeader

                if let selectedLane {
                    conversationContext(for: selectedLane)
                    Divider().opacity(0.36)
                }

                transcript

                Divider().opacity(0.48)
                composer
            }
        }
        .accessibilityIdentifier("loom-conversation-space")
    }

    private var conversationHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            Circle()
                .fill(accent)
                .frame(width: 10, height: 10)
                .shadow(color: accent.opacity(0.8), radius: 7)

            VStack(alignment: .leading, spacing: 2) {
                Text(selectedLane?.agent ?? "CONVERSATION")
                    .font(.system(size: 18, weight: .semibold, design: .rounded))
                Text(selectedLane.map { "\($0.lane) · \($0.deliveryReadinessLabel)" } ?? "Select a live lane to continue")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 12)

            conversationStatus
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
    }

    private var conversationStatus: some View {
        HStack(spacing: 6) {
            Image(systemName: statusSymbol)
            Text(statusLabel)
        }
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .foregroundStyle(statusColor)
        .padding(.horizontal, 10)
        .frame(height: 26)
        .background(statusColor.opacity(0.10), in: Capsule())
        .overlay(Capsule().stroke(statusColor.opacity(0.24), lineWidth: 0.8))
    }

    private var transcript: some View {
        ScrollViewReader { scrollProxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 15) {
                    timelineHeader

                    if let error = store.threadError {
                        ConversationStateCard(
                            title: "THREAD READ REFUSED",
                            detail: error,
                            color: LoomColor.red
                        )
                    } else if store.visibleThreadEvents.isEmpty {
                        ConversationEmptyState(
                            title: store.messageBridgeConfigured ? "Start where you are" : "Message bridge unavailable",
                            detail: emptyStateDetail,
                            color: selectedLane?.deliveryReadiness == .immediate ? LoomColor.green : LoomColor.amber
                        )
                    } else {
                        ForEach(store.visibleThreadEvents) { event in
                            ConversationEventBubble(event: event)
                                .id(event.id)
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 18)
            }
            .onChange(of: store.visibleThreadEvents.last?.id) { _, eventID in
                guard let eventID else { return }
                scrollProxy.scrollTo(eventID, anchor: .bottom)
            }
        }
    }

    private var timelineHeader: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(store.visibleThreadState == "active" ? LoomColor.green : LoomColor.cyan)
                .frame(width: 6, height: 6)
            Text("CONTINUOUS THREAD")
            Text(store.visibleThreadEvents.isEmpty ? "READY" : "\(store.visibleThreadEvents.count) TURNS")
                .foregroundStyle(.secondary)
            Spacer()
            if let thread = store.selectedThread {
                Text(thread.request.id)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .foregroundStyle(LoomColor.cyan)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .bottom, spacing: 10) {
                TextField(
                    selectedLane.map { "Continue with \($0.agent)…" } ?? "Select a lane to begin",
                    text: $store.conversationDraft,
                    axis: .vertical
                )
                .textFieldStyle(.plain)
                .font(.system(size: 15))
                .lineLimit(1...7)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .background(Color.black.opacity(0.28), in: RoundedRectangle(cornerRadius: 13, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .stroke(accent.opacity(0.28), lineWidth: 0.9)
                )
                .accessibilityIdentifier("loom-conversation-draft")
                .accessibilityLabel("Continue the selected conversation")

                Button {
                    Task { await store.sendMessage() }
                } label: {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 17, weight: .bold))
                        .foregroundStyle(store.canSendMessage ? LoomColor.ink : .secondary)
                        .frame(width: 44, height: 44)
                        .background(store.canSendMessage ? LoomColor.cyan : Color.white.opacity(0.10), in: Circle())
                        .shadow(color: store.canSendMessage ? LoomColor.cyan.opacity(0.30) : .clear, radius: 12, y: 4)
                }
                .buttonStyle(.plain)
                .disabled(!store.canSendMessage)
                .keyboardShortcut(.return, modifiers: [.command])
                .help("Send through the authenticated Loom message bridge")
                .accessibilityIdentifier("loom-conversation-send")
                .accessibilityLabel("Send durable message")
            }

            HStack(spacing: 6) {
                Image(systemName: statusSymbol)
                Text(statusLabel)
                Spacer(minLength: 8)
                if let selectedLane {
                    Text("to \(selectedLane.agent)")
                }
            }
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .foregroundStyle(statusColor)
        }
        .padding(14)
    }

    private func conversationContext(for lane: LoomFleetSnapshot.Lane) -> some View {
        HStack(spacing: 16) {
            contextField("PRESENCE", lane.displayState.uppercased(), lane.displayColor)
            contextField("DELIVERY", lane.deliveryReadinessLabel, lane.deliveryReadiness == .immediate ? LoomColor.green : LoomColor.amber)
            contextField("HARNESS", lane.harness.uppercased(), LoomColor.cyan)
            contextField("CURSOR", String(lane.cursor), LoomColor.magenta)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 18)
        .padding(.bottom, 12)
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

    private var emptyStateDetail: String {
        guard let selectedLane else {
            return "A live lane will appear here as soon as the fleet publishes it."
        }
        if selectedLane.deliveryReadiness == .immediate {
            return "This agent is present. Say what you need in ordinary language; the durable bus carries the receipt underneath."
        }
        return "The message remains durable while this agent is away. The conversation is waiting, not lost."
    }

    private var statusColor: Color {
        switch store.messageState {
        case .ready: selectedLane?.deliveryReadiness == .immediate ? LoomColor.green : LoomColor.amber
        case .accepted: LoomColor.green
        case .sending: LoomColor.cyan
        case .failed: LoomColor.red
        case .unconfigured: LoomColor.amber
        }
    }

    private var statusLabel: String {
        switch store.messageState {
        case .ready:
            selectedLane?.deliveryReadiness == .immediate ? "LIVE ENDPOINT" : "DURABLE DELIVERY"
        case .sending: "SENDING DURABLE MESSAGE"
        case .accepted: "RECEIPT STORED"
        case .failed: "DELIVERY REFUSED"
        case .unconfigured: "BRIDGE UNAVAILABLE"
        }
    }

    private var statusSymbol: String {
        switch store.messageState {
        case .ready: selectedLane?.deliveryReadiness == .immediate ? "antenna.radiowaves.left.and.right" : "archivebox.fill"
        case .sending: "arrow.trianglehead.2.clockwise.rotate.90"
        case .accepted: "checkmark.seal.fill"
        case .failed: "exclamationmark.octagon.fill"
        case .unconfigured: "cable.connector.slash"
        }
    }
}

struct ConversationContextDrawer: View {
    @ObservedObject var store: LoomStore

    var body: some View {
        GlassSurface(radius: 16) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("CONTEXT")
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                        Text("Quietly attached to this thread")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: "slider.horizontal.3")
                        .foregroundStyle(LoomColor.cyan)
                }

                ContextCard(
                    title: "WORKING WITH",
                    detail: store.selectedLane?.agent ?? "No selected lane",
                    accent: LoomColor.cyan
                ) {
                    Text(store.selectedLane?.lane ?? "Select a live lane to attach context")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                ContextCard(
                    title: "ROUTE RECEIPT",
                    detail: store.dashboard.receipt.status.rawValue.uppercased(),
                    accent: store.dashboard.receipt.status.loomColor
                ) {
                    ReceiptLine(label: "task", value: store.dashboard.receipt.taskId)
                    ReceiptLine(label: "policy", value: store.dashboard.receipt.policy)
                    ReceiptLine(label: "model", value: store.dashboard.receipt.model)
                    ReceiptLine(label: "effort", value: store.dashboard.receipt.effort)
                }

                ContextCard(
                    title: "FLEET SIGNAL",
                    detail: fleetDetail,
                    accent: fleetColor
                ) {
                    Text("The backend keeps the decision. This view keeps the surrounding evidence close.")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: 0)

                Label("System activity stays out of the conversation unless it changes your next move.", systemImage: "eye.slash")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(14)
        }
        .accessibilityIdentifier("loom-conversation-context")
    }

    private var fleetDetail: String {
        guard let fleet = store.fleet else { return "Awaiting fleet" }
        return "\(fleet.summary.live)/\(fleet.summary.lanes) live"
    }

    private var fleetColor: Color {
        store.fleet?.coordinationAvailable == true ? LoomColor.green : LoomColor.amber
    }
}

private struct ConversationEventBubble: View {
    let event: LoomThreadEvent

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            if event.isLocal { Spacer(minLength: 80) }

            VStack(alignment: event.isLocal ? .trailing : .leading, spacing: 6) {
                HStack(spacing: 7) {
                    if !event.isLocal {
                        Circle()
                            .fill(LoomColor.magenta)
                            .frame(width: 8, height: 8)
                            .shadow(color: LoomColor.magenta.opacity(0.7), radius: 5)
                    }
                    Text(event.isLocal ? "YOU" : displayActor)
                    Text(shortTime).foregroundStyle(.secondary)
                }
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(event.isLocal ? LoomColor.cyan : LoomColor.magenta)

                Text(event.body)
                    .font(.system(size: 14))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                    .background(
                        (event.isLocal ? LoomColor.cyan : Color.white).opacity(event.isLocal ? 0.14 : 0.075),
                        in: RoundedRectangle(cornerRadius: 15, style: .continuous)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 15, style: .continuous)
                            .stroke((event.isLocal ? LoomColor.cyan : Color.white).opacity(0.18), lineWidth: 0.8)
                    )
            }
            .frame(maxWidth: 590, alignment: event.isLocal ? .trailing : .leading)

            if !event.isLocal { Spacer(minLength: 80) }
        }
    }

    private var shortTime: String {
        guard let marker = event.utc.lastIndex(of: "T") else { return event.utc }
        return String(event.utc[event.utc.index(after: marker)...].prefix(8))
    }

    private var displayActor: String {
        event.actor.split(separator: "/", maxSplits: 1).first
            .map { String($0).uppercased() } ?? event.actor.uppercased()
    }
}

private struct ConversationEmptyState: View {
    let title: String
    let detail: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right.fill")
                .font(.system(size: 24, weight: .medium))
                .foregroundStyle(color)
            Text(title)
                .font(.system(size: 17, weight: .semibold, design: .rounded))
            Text(detail)
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .background(color.opacity(0.06), in: RoundedRectangle(cornerRadius: 15, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 15, style: .continuous).stroke(color.opacity(0.22)))
    }
}

private struct ConversationStateCard: View {
    let title: String
    let detail: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(color)
            Text(detail)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(color.opacity(0.07), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(color.opacity(0.28)))
    }
}

private struct ContextCard<Content: View>: View {
    let title: String
    let detail: String
    let accent: Color
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(accent)
            Text(detail)
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(2)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.black.opacity(0.22), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color.white.opacity(0.10)))
    }
}

private struct ReceiptLine: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 42, alignment: .leading)
            Text(value)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .font(.system(size: 9, weight: .medium, design: .monospaced))
    }
}

private extension LoomFleetSnapshot.Lane {
    var displayState: String {
        if !presenceState.isEmpty && presenceState != "none" { return presenceState }
        return state
    }

    var displayColor: Color {
        switch displayState {
        case "live", "active": LoomColor.green
        case "claimed", "recoverable": LoomColor.amber
        case "unresponsive", "orphaned", "lost": LoomColor.red
        default: .secondary
        }
    }

    var deliveryReadinessLabel: String {
        deliveryReadiness == .immediate ? "LIVE" : "DURABLE"
    }
}

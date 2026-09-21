import Foundation
import Combine
import LoomDomain

@MainActor
final class LoomStore: ObservableObject {
    enum ConnectionState: Equatable {
        case connecting
        case connected
        case unavailable(String)

        var label: String {
            switch self {
            case .connecting: "CONNECTING"
            case .connected: "LIVE / READ ONLY"
            case .unavailable: "MOCK / OFFLINE"
            }
        }
    }

    enum MessageBridgeState: Equatable {
        case unconfigured
        case ready
        case sending
        case accepted(LoomMessageReceipt)
        case failed(String)

        var label: String {
            switch self {
            case .unconfigured: "MESSAGE BRIDGE NOT CONFIGURED"
            case .ready: "DURABLE BUS READY"
            case .sending: "REQUESTING DURABLE RECEIPT"
            case let .accepted(receipt):
                receipt.wakeStatus == "durable_only"
                    ? "DURABLE / WAKE UNAVAILABLE"
                    : "DURABLE / DELIVERY ATTEMPTED"
            case .failed: "MESSAGE REFUSED"
            }
        }
    }

    @Published var scenario: DashboardScenario = .nominal {
        didSet { dashboard = .mock(scenario) }
    }
    @Published private(set) var dashboard = DashboardSnapshot.mock(.nominal)
    @Published private(set) var fleet: LoomFleetSnapshot?
    @Published private(set) var eventGroups: [LoomEventGroup] = []
    @Published private(set) var connection: ConnectionState = .connecting
    @Published private(set) var messageState: MessageBridgeState
    @Published var selectedLaneId: String?
    @Published var conversationDraft = ""

    private let client: LoomFleetClient
    private let messageClient: LoomMessageClient?

    var selectedLane: LoomFleetSnapshot.Lane? {
        guard let selectedLaneId else { return nil }
        return fleet?.lanes.first { $0.id == selectedLaneId }
    }

    var canSendMessage: Bool {
        messageClient != nil
            && selectedLane != nil
            && !conversationDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && messageState != .sending
    }

    var messageBridgeConfigured: Bool { messageClient != nil }

    init(
        baseURL: URL = URL(string: "http://127.0.0.1:8787")!,
        arguments: [String] = ProcessInfo.processInfo.arguments
    ) {
        client = LoomFleetClient(baseURL: baseURL)
        let messageURL = Self.argument("--message-url", in: arguments)
        let tokenPath = Self.argument("--message-token-file", in: arguments)
        if let messageURL, let tokenPath, let url = URL(string: messageURL),
           let rawToken = try? String(contentsOfFile: tokenPath, encoding: .utf8)
        {
            let token = rawToken.trimmingCharacters(in: .whitespacesAndNewlines)
            if token.count >= 32 {
                messageClient = LoomMessageClient(baseURL: url, capability: token)
                messageState = .ready
            } else {
                messageClient = nil
                messageState = .failed("Capability file is invalid")
            }
        } else if messageURL == nil && tokenPath == nil {
            messageClient = nil
            messageState = .unconfigured
        } else {
            messageClient = nil
            messageState = .failed("Message bridge configuration is incomplete")
        }
    }

    func refreshFleet() async {
        do {
            let snapshot = try await client.fleet()
            fleet = snapshot
            connection = .connected
            if selectedLaneId == nil || !snapshot.lanes.contains(where: { $0.id == selectedLaneId }) {
                selectedLaneId = snapshot.lanes.first(where: { $0.presenceState == "live" })?.id
                    ?? snapshot.lanes.first(where: { $0.state == "live" || $0.state == "active" })?.id
                    ?? snapshot.lanes.first?.id
            }
            if let events = try? await client.events() {
                eventGroups = events
            }
        } catch {
            connection = .unavailable(error.localizedDescription)
        }
    }

    func poll() async {
        while !Task.isCancelled {
            await refreshFleet()
            try? await Task.sleep(for: .seconds(4))
        }
    }

    func sendMessage() async {
        guard let messageClient, let lane = selectedLane else { return }
        let message = conversationDraft
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }

        messageState = .sending
        do {
            let receipt = try await messageClient.send(
                LoomMessageRequest(toAgent: lane.agent, toLane: lane.lane, message: message)
            )
            conversationDraft = ""
            messageState = .accepted(receipt)
        } catch {
            messageState = .failed(error.localizedDescription)
        }
    }

    private static func argument(_ name: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: name),
              arguments.indices.contains(index + 1)
        else { return nil }
        return arguments[index + 1]
    }
}

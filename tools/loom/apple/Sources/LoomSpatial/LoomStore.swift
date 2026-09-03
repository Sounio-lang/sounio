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
    @Published private(set) var threadRequests: [LoomBusMessage] = []
    @Published private(set) var selectedThread: LoomMessageThread?
    @Published private(set) var threadError: String?
    @Published var selectedLaneId: String? {
        didSet {
            guard oldValue != selectedLaneId else { return }
            Task { await refreshThreads() }
        }
    }
    @Published var conversationDraft = ""

    private let client: LoomFleetClient
    private let messageClient: LoomMessageClient?
    private var selectedThreadID: String?
    private var refreshingThreads = false

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

    var visibleThreadEvents: [LoomThreadEvent] { selectedThread?.events ?? [] }

    var visibleThreadState: String? { selectedThread?.state }

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
                selectedLaneId = LoomFleetSnapshot.Lane.preferredDeliveryLane(in: snapshot.lanes)?.id
            }
            if let events = try? await client.events() {
                eventGroups = events
            }
            await refreshThreads()
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
            selectedThreadID = receipt.messageId
            await refreshThreads()
        } catch {
            messageState = .failed(error.localizedDescription)
        }
    }

    func refreshThreads() async {
        guard let messageClient, !refreshingThreads else { return }
        refreshingThreads = true
        defer { refreshingThreads = false }
        do {
            let list = try await messageClient.threads()
            threadRequests = list.threads
            threadError = nil
            let laneThreads = list.threads.filter { thread in
                guard let lane = selectedLane else { return false }
                return thread.toAgent == lane.agent && thread.toLane == lane.lane
            }
            if let selectedThreadID,
               laneThreads.contains(where: { $0.id == selectedThreadID }) == false
            {
                self.selectedThreadID = laneThreads.first?.id
            } else if selectedThreadID == nil {
                selectedThreadID = laneThreads.first?.id
            }
            guard let selectedThreadID else {
                selectedThread = nil
                return
            }
            try await refreshThread(selectedThreadID, using: messageClient)
        } catch {
            threadError = error.localizedDescription
        }
    }

    private func refreshThread(
        _ messageID: String,
        using messageClient: LoomMessageClient
    ) async throws {
        let firstRead = try await messageClient.thread(messageID)
        selectedThread = firstRead
        let acknowledged = Set(
            firstRead.events
                .filter { $0.kind == "ack" }
                .map(\.messageId)
        )
        let pendingAcknowledgements = firstRead.events
            .filter { $0.kind == "response" && !acknowledged.contains($0.messageId) }
            .map(\.messageId)
        guard pendingAcknowledgements.isEmpty == false else { return }
        for responseID in pendingAcknowledgements {
            _ = try await messageClient.acknowledge(responseID)
        }
        selectedThread = try await messageClient.thread(messageID)
    }

    private static func argument(_ name: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: name),
              arguments.indices.contains(index + 1)
        else { return nil }
        return arguments[index + 1]
    }
}

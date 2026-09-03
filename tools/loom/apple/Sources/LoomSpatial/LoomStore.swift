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

    enum RoutingConfigState: Equatable {
        case unconfigured
        case loading
        case ready
        case editing
        case saving
        case stored(LoomRoutingConfigReceipt)
        case failed(String)

        var label: String {
            switch self {
            case .unconfigured: "ROUTING BRIDGE NOT CONFIGURED"
            case .loading: "LOADING DECLARATIVE CONFIG"
            case .ready: "CONFIG READY / BACKEND ARBITRATES"
            case .editing: "CONFIGURATION CHANGED / NOT STORED"
            case .saving: "STORING CONFIGURATION"
            case .stored: "CONFIGURATION STORED"
            case .failed: "CONFIGURATION REFUSED"
            }
        }
    }

    enum RouteState: Equatable {
        case ready
        case deciding
        case received(LoomRouteOperation)
        case failed(String)

        var label: String {
            switch self {
            case .ready: "READY FOR SOUNIO DECISION"
            case .deciding: "SOUNIO 9032 DECIDING"
            case let .received(operation):
                "RECEIPT \(operation.receipt.status.rawValue.uppercased())"
            case .failed: "ROUTE REQUEST FAILED"
            }
        }
    }

    @Published var scenario: DashboardScenario = .nominal {
        didSet {
            routeOperation = nil
            routeState = .ready
            dashboard = .mock(scenario)
        }
    }
    @Published private(set) var dashboard = DashboardSnapshot.mock(.nominal)
    @Published private(set) var fleet: LoomFleetSnapshot?
    @Published private(set) var eventGroups: [LoomEventGroup] = []
    @Published private(set) var connection: ConnectionState = .connecting
    @Published private(set) var messageState: MessageBridgeState
    @Published private(set) var threadRequests: [LoomBusMessage] = []
    @Published private(set) var selectedThread: LoomMessageThread?
    @Published private(set) var threadError: String?
    @Published private(set) var routingConfigState: RoutingConfigState
    @Published private(set) var routingConfig: LoomRoutingConfig?
    @Published private(set) var routeState: RouteState = .ready
    @Published private(set) var routeOperation: LoomRouteOperation?
    @Published private(set) var routingDraftDirty = false
    @Published var routingDraft = LoomRoutingConfigUpdate(
        policy: "authority-first",
        model: "gpt-5.6-terra",
        effort: "high",
        poolOrder: ["pool-openai-team"],
        adapterOrder: ["adapter-codex"]
    )
    @Published var selectedLaneId: String? {
        didSet {
            guard oldValue != selectedLaneId else { return }
            Task { await refreshThreads() }
        }
    }
    @Published var conversationDraft = ""
    @Published var routeTitle = "Review current Loom evidence"
    @Published var routePrompt = "Identify the highest operational risk in the current routing evidence. Do not change files."

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

    var canSaveRoutingConfig: Bool {
        guard messageClient != nil else { return false }
        guard case .saving = routingConfigState else { return true }
        return false
    }

    var canRouteTask: Bool {
        guard messageClient != nil, routingDraftDirty == false else { return false }
        guard case .deciding = routeState else {
            return !routeTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && !routePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return false
    }

    var dashboardIsLive: Bool { routeOperation != nil }

    var visibleThreadEvents: [LoomThreadEvent] { selectedThread?.events ?? [] }

    var visibleThreadState: String? { selectedThread?.state }

    init(
        baseURL: URL = URL(string: "http://127.0.0.1:8787")!,
        arguments: [String] = ProcessInfo.processInfo.arguments
    ) {
        let kernelURL = Self.argument("--kernel-url", in: arguments)
            .flatMap(URL.init(string:)) ?? baseURL
        client = LoomFleetClient(baseURL: kernelURL)
        let messageURL = Self.argument("--message-url", in: arguments)
        let tokenPath = Self.argument("--message-token-file", in: arguments)
        if let messageURL, let tokenPath, let url = URL(string: messageURL),
           let rawToken = try? String(contentsOfFile: tokenPath, encoding: .utf8)
        {
            let token = rawToken.trimmingCharacters(in: .whitespacesAndNewlines)
            if token.count >= 32 {
                messageClient = LoomMessageClient(baseURL: url, capability: token)
                messageState = .ready
                routingConfigState = .loading
            } else {
                messageClient = nil
                messageState = .failed("Capability file is invalid")
                routingConfigState = .unconfigured
            }
        } else if messageURL == nil && tokenPath == nil {
                messageClient = nil
                messageState = .unconfigured
                routingConfigState = .unconfigured
        } else {
                messageClient = nil
                messageState = .failed("Message bridge configuration is incomplete")
                routingConfigState = .unconfigured
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
            await refreshRoutingConfig()
            await refreshLatestRouteOperation()
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

    func refreshRoutingConfig() async {
        guard let messageClient else {
            routingConfigState = .unconfigured
            return
        }
        if case .saving = routingConfigState { return }
        if routingConfig == nil { routingConfigState = .loading }
        do {
            let config = try await messageClient.routingConfig()
            routingConfig = config
            if routingDraftDirty == false {
                routingDraft = config.update
            }
            switch routingConfigState {
            case .stored, .editing:
                // Preserve the local storage receipt or a pending local draft.
                break
            default:
                routingConfigState = .ready
            }
        } catch {
            routingConfigState = .failed(error.localizedDescription)
        }
    }

    func refreshLatestRouteOperation() async {
        guard let messageClient, routeState != .deciding else { return }
        do {
            guard let operation = try await messageClient.latestRouteOperation() else { return }
            routeOperation = operation
            dashboard = .live(operation, title: operation.receipt.taskId)
            routeState = .received(operation)
        } catch {
            // Fleet and configuration remain usable when no historical receipt is readable.
        }
    }

    func saveRoutingConfig() async {
        guard let messageClient else { return }
        routingConfigState = .saving
        do {
            let receipt = try await messageClient.updateRoutingConfig(routingDraft)
            routingConfig = receipt.config
            routingDraft = receipt.config.update
            routingDraftDirty = false
            routingConfigState = .stored(receipt)
        } catch {
            routingConfigState = .failed(error.localizedDescription)
        }
    }

    func routeTask() async {
        guard let messageClient, canRouteTask else { return }
        let title = routeTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        let prompt = routePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let taskID = "ui-" + UUID().uuidString.lowercased()
        routeState = .deciding
        do {
            let operation = try await messageClient.route(
                LoomRouteTaskRequest(taskId: taskID, title: title, prompt: prompt)
            )
            routeOperation = operation
            dashboard = .live(operation, title: title)
            routeState = .received(operation)
        } catch {
            routeState = .failed(error.localizedDescription)
        }
    }

    func setRoutingPolicy(_ policy: String) {
        routingDraft.policy = policy
        markRoutingDraftEdited()
    }

    func setRoutingModel(_ model: String) {
        routingDraft.model = model
        markRoutingDraftEdited()
    }

    func setRoutingEffort(_ effort: String) {
        routingDraft.effort = effort
        markRoutingDraftEdited()
    }

    private func markRoutingDraftEdited() {
        routingDraftDirty = true
        routingConfigState = .editing
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

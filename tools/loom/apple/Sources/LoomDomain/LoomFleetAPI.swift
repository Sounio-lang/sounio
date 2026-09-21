import Foundation

public struct LoomFleetSnapshot: Codable, Equatable, Sendable {
    public struct Summary: Codable, Equatable, Sendable {
        public let lanes: Int
        public let live: Int
        public let unresponsive: Int
        public let orphaned: Int
        public let loomCustody: Int
        public let activeEndpoints: Int

        enum CodingKeys: String, CodingKey {
            case lanes, live, unresponsive, orphaned
            case loomCustody = "loom_custody"
            case activeEndpoints = "active_endpoints"
        }
    }

    public struct Lane: Identifiable, Codable, Equatable, Sendable {
        public var id: String { agent + "/" + lane }

        public let agent: String
        public let lane: String
        public let state: String
        public let claimState: String
        public let presenceState: String
        public let endpointState: String
        public let harness: String
        public let worktree: String
        public let loomState: String
        public let cursor: Int

        public init(
            agent: String,
            lane: String,
            state: String,
            claimState: String,
            presenceState: String,
            endpointState: String,
            harness: String,
            worktree: String,
            loomState: String,
            cursor: Int
        ) {
            self.agent = agent
            self.lane = lane
            self.state = state
            self.claimState = claimState
            self.presenceState = presenceState
            self.endpointState = endpointState
            self.harness = harness
            self.worktree = worktree
            self.loomState = loomState
            self.cursor = cursor
        }

        public var hasLivePresence: Bool {
            presenceState == "live" || presenceState == "active"
        }

        public var hasActiveEndpoint: Bool { endpointState == "active" }

        public var deliveryReadiness: LaneDeliveryReadiness {
            hasActiveEndpoint ? .immediate : .durableOnly
        }

        public static func preferredDeliveryLane(in lanes: [Self]) -> Self? {
            lanes.first { $0.hasLivePresence && $0.hasActiveEndpoint }
                ?? lanes.first { $0.hasLivePresence }
                ?? lanes.first { $0.hasActiveEndpoint }
                ?? lanes.first
        }

        enum CodingKeys: String, CodingKey {
            case agent, lane, state, harness, worktree, cursor
            case claimState = "claim_state"
            case presenceState = "presence_state"
            case endpointState = "endpoint_state"
            case loomState = "loom_state"
        }
    }

    public let schema: String
    public let snapshotUTC: String
    public let coordinationAvailable: Bool
    public let summary: Summary
    public let lanes: [Lane]

    enum CodingKeys: String, CodingKey {
        case schema, summary, lanes
        case snapshotUTC = "snapshot_utc"
        case coordinationAvailable = "coordination_available"
    }
}

public enum LaneDeliveryReadiness: String, Codable, Equatable, Sendable {
    case immediate = "immediate"
    case durableOnly = "durable_only"
}

public struct LoomEventGroup: Identifiable, Codable, Equatable, Sendable {
    public struct Event: Identifiable, Codable, Equatable, Sendable {
        public var id: String { source + ":" + String(sequence) + ":" + hash }

        public let source: String
        public let sequence: Int
        public let utc: String
        public let kind: String
        public let hash: String

        enum CodingKeys: String, CodingKey {
            case source, utc, kind, hash
            case sequence = "seq"
        }
    }

    public var id: String { agent + "/" + lane + "/" + instanceId }

    public let agent: String
    public let lane: String
    public let instanceId: String
    public let state: String
    public let verified: Bool
    public let journalProfile: String
    public let recoveries: Int
    public let semanticHead: String
    public let guardianHead: String
    public let events: [Event]

    enum CodingKeys: String, CodingKey {
        case agent, lane, state, verified, recoveries, events
        case instanceId = "instance_id"
        case journalProfile = "journal_profile"
        case semanticHead = "semantic_head"
        case guardianHead = "guardian_head"
    }
}

public struct LoomFleetClient: Sendable {
    public let baseURL: URL

    public init(baseURL: URL) {
        self.baseURL = baseURL
    }

    public func fleet() async throws -> LoomFleetSnapshot {
        let url = baseURL.appending(path: "api/fleet")
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 12
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(LoomFleetSnapshot.self, from: data)
    }

    public func events() async throws -> [LoomEventGroup] {
        let url = baseURL.appending(path: "api/events")
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 8
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode([LoomEventGroup].self, from: data)
    }

    public func snapshot(agent: String, lane: String, cursor: Int) async throws -> (Data, Int) {
        var components = URLComponents(
            url: baseURL.appending(path: "api/snapshot"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "agent", value: agent),
            URLQueryItem(name: "lane", value: lane),
            URLQueryItem(name: "cursor", value: String(cursor)),
        ]
        guard let url = components?.url else { throw URLError(.badURL) }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        let next = Int(http.value(forHTTPHeaderField: "X-Loom-Cursor") ?? "") ?? cursor
        return (data, next)
    }
}

public struct LoomMessageRequest: Codable, Equatable, Sendable {
    public let toAgent: String
    public let toLane: String
    public let kind: String
    public let message: String

    public init(toAgent: String, toLane: String, kind: String = "request", message: String) {
        self.toAgent = toAgent
        self.toLane = toLane
        self.kind = kind
        self.message = message
    }
}

public struct LoomMessageReceipt: Codable, Equatable, Sendable {
    public let schema: String
    public let messageId: String
    public let threadId: String
    public let toAgent: String
    public let toLane: String
    public let kind: String
    public let status: String
    public let wakeStatus: String
}

public struct LoomBusMessage: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let utc: String
    public let createdEpoch: Int
    public let fromAgent: String
    public let fromLane: String
    public let toAgent: String
    public let toLane: String
    public let kind: String
    public let text: String
    public let threadId: String
    public let replyTo: String
}

public struct LoomThreadEvent: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let utc: String
    public let kind: String
    public let state: String
    public let messageId: String
    public let actor: String
    public let body: String

    public var isLocal: Bool { kind == "request" || kind == "ack" }
}

public struct LoomMessageThread: Codable, Equatable, Sendable {
    public let schema: String
    public let request: LoomBusMessage
    public let state: String
    public let delivery: String
    public let injected: Int
    public let acknowledged: Int
    public let responseCount: Int
    public let wakeCount: Int
    public let wakePending: Int
    public let timeoutSeconds: Int
    public let events: [LoomThreadEvent]
}

public struct LoomMessageThreadList: Codable, Equatable, Sendable {
    public let schema: String
    public let senderAgent: String
    public let senderLane: String
    public let threads: [LoomBusMessage]
}

public struct LoomMessageAcknowledgement: Codable, Equatable, Sendable {
    public let schema: String
    public let messageId: String
    public let status: String
}

public struct LoomRoutingConfig: Codable, Equatable, Sendable {
    public let schema: String
    public let revision: Int
    public let updatedEpoch: Int
    public let policy: String
    public let model: String
    public let effort: String
    public let poolOrder: [String]
    public let adapterOrder: [String]

    public var update: LoomRoutingConfigUpdate {
        LoomRoutingConfigUpdate(
            schema: schema,
            policy: policy,
            model: model,
            effort: effort,
            poolOrder: poolOrder,
            adapterOrder: adapterOrder
        )
    }
}

public struct LoomRoutingConfigUpdate: Codable, Equatable, Sendable {
    public let schema: String
    public var policy: String
    public var model: String
    public var effort: String
    public var poolOrder: [String]
    public var adapterOrder: [String]

    public init(
        schema: String = "loom-routing-config-v1",
        policy: String,
        model: String,
        effort: String,
        poolOrder: [String],
        adapterOrder: [String]
    ) {
        self.schema = schema
        self.policy = policy
        self.model = model
        self.effort = effort
        self.poolOrder = poolOrder
        self.adapterOrder = adapterOrder
    }
}

public struct LoomRoutingConfigReceipt: Codable, Equatable, Sendable {
    public let schema: String
    public let revision: Int
    public let updatedEpoch: Int
    public let previousDigest: String
    public let digest: String
    public let status: String
    public let config: LoomRoutingConfig
}

public struct LoomLatestRouteOperation: Codable, Equatable, Sendable {
    public let schema: String
    public let operation: LoomRouteOperation?
}

public enum LoomMessageClientError: LocalizedError, Equatable, Sendable {
    case refused(status: Int, reason: String)
    case invalidReceipt

    public var errorDescription: String? {
        switch self {
        case let .refused(status, reason): "Bridge refused (\(status)): \(reason)"
        case .invalidReceipt: "Bridge returned an invalid receipt"
        }
    }
}

public struct LoomMessageClient: Sendable {
    private struct ErrorEnvelope: Decodable {
        let error: String
    }

    public let baseURL: URL
    private let capability: String

    public init(baseURL: URL, capability: String) {
        self.baseURL = baseURL
        self.capability = capability
    }

    private func authorizedRequest(path: String, method: String = "GET") -> URLRequest {
        let url = baseURL.appending(path: path)
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 12
        request.setValue("Bearer \(capability)", forHTTPHeaderField: "Authorization")
        return request
    }

    private func authorizedRequest(
        path: String,
        method: String,
        timeout: TimeInterval
    ) -> URLRequest {
        var request = authorizedRequest(path: path, method: method)
        request.timeoutInterval = timeout
        return request
    }

    private func checkedResponse<T: Decodable>(
        _ request: URLRequest,
        expectedStatus: Int = 200,
        as type: T.Type
    ) async throws -> T {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw LoomMessageClientError.invalidReceipt
        }
        guard http.statusCode == expectedStatus else {
            let reason = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data).error)
                ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw LoomMessageClientError.refused(status: http.statusCode, reason: reason)
        }
        return try JSONDecoder().decode(type, from: data)
    }

    public func send(_ message: LoomMessageRequest) async throws -> LoomMessageReceipt {
        var request = authorizedRequest(path: "v1/messages", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(message)
        let receipt = try await checkedResponse(request, expectedStatus: 202, as: LoomMessageReceipt.self)
        guard receipt.schema == "loom-message-receipt-v1",
              receipt.status == "accepted",
              receipt.messageId.isEmpty == false,
              receipt.threadId.isEmpty == false
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return receipt
    }

    public func threads(limit: Int = 40) async throws -> LoomMessageThreadList {
        let bounded = min(max(limit, 1), 100)
        var components = URLComponents(url: baseURL.appending(path: "v1/threads"), resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "limit", value: String(bounded))]
        guard let url = components?.url else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 12
        request.setValue("Bearer \(capability)", forHTTPHeaderField: "Authorization")
        let list = try await checkedResponse(request, as: LoomMessageThreadList.self)
        guard list.schema == "loom-message-thread-list-v1" else {
            throw LoomMessageClientError.invalidReceipt
        }
        return list
    }

    public func thread(_ messageID: String, timeoutSeconds: Int = 60) async throws -> LoomMessageThread {
        let bounded = min(max(timeoutSeconds, 0), 86_400)
        var components = URLComponents(
            url: baseURL.appending(path: "v1/threads/\(messageID)"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [URLQueryItem(name: "timeoutSeconds", value: String(bounded))]
        guard let url = components?.url else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 12
        request.setValue("Bearer \(capability)", forHTTPHeaderField: "Authorization")
        let thread = try await checkedResponse(request, as: LoomMessageThread.self)
        guard thread.schema == "loom-message-thread-v1", thread.request.id == messageID else {
            throw LoomMessageClientError.invalidReceipt
        }
        return thread
    }

    public func acknowledge(_ messageID: String) async throws -> LoomMessageAcknowledgement {
        let receipt = try await checkedResponse(
            authorizedRequest(path: "v1/messages/\(messageID)/ack", method: "POST"),
            as: LoomMessageAcknowledgement.self
        )
        guard receipt.schema == "loom-message-ack-v1",
              receipt.messageId == messageID,
              receipt.status == "acknowledged"
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return receipt
    }

    public func routingConfig() async throws -> LoomRoutingConfig {
        let config = try await checkedResponse(
            authorizedRequest(path: "v1/routing/config"),
            as: LoomRoutingConfig.self
        )
        guard config.schema == "loom-routing-config-v1",
              config.revision >= 0,
              config.poolOrder.isEmpty == false,
              config.adapterOrder.isEmpty == false
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return config
    }

    public func latestRouteOperation() async throws -> LoomRouteOperation? {
        let latest = try await checkedResponse(
            authorizedRequest(path: "v1/routing/receipts/latest"),
            as: LoomLatestRouteOperation.self
        )
        guard latest.schema == "loom-latest-route-operation-v1" else {
            throw LoomMessageClientError.invalidReceipt
        }
        if let operation = latest.operation {
            guard operation.schema == "loom-route-operation-v1",
                  operation.decision.taskId == operation.receipt.taskId,
                  operation.receipt.producingLanguage == "Sounio",
                  operation.receipt.languageRole == "SEMANTIC_AUTHORITY"
            else {
                throw LoomMessageClientError.invalidReceipt
            }
        }
        return latest.operation
    }

    public func updateRoutingConfig(
        _ update: LoomRoutingConfigUpdate
    ) async throws -> LoomRoutingConfigReceipt {
        var request = authorizedRequest(path: "v1/routing/config", method: "PUT")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(update)
        let receipt = try await checkedResponse(request, as: LoomRoutingConfigReceipt.self)
        guard receipt.schema == "loom-routing-config-receipt-v1",
              receipt.status == "stored",
              receipt.config.schema == "loom-routing-config-v1",
              receipt.config.revision == receipt.revision,
              receipt.digest.isEmpty == false,
              receipt.previousDigest.isEmpty == false
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return receipt
    }

    public func route(_ task: LoomRouteTaskRequest) async throws -> LoomRouteOperation {
        var request = authorizedRequest(
            path: "v1/routing/tasks",
            method: "POST",
            timeout: 75
        )
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(task)
        let operation = try await checkedResponse(request, as: LoomRouteOperation.self)
        guard operation.schema == "loom-route-operation-v1",
              operation.decision.taskId == task.taskId,
              operation.receipt.taskId == task.taskId,
              operation.receipt.producingLanguage == "Sounio",
              operation.receipt.languageRole == "SEMANTIC_AUTHORITY"
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return operation
    }

    public func routeOperation(taskID: String) async throws -> LoomRouteOperation {
        let operation = try await checkedResponse(
            authorizedRequest(path: "v1/routing/tasks/\(taskID)"),
            as: LoomRouteOperation.self
        )
        guard operation.schema == "loom-route-operation-v1",
              operation.decision.taskId == taskID,
              operation.receipt.taskId == taskID,
              operation.receipt.producingLanguage == "Sounio",
              operation.receipt.languageRole == "SEMANTIC_AUTHORITY"
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return operation
    }

    public func cancelRoute(taskID: String) async throws -> LoomRouteOperation {
        let operation = try await checkedResponse(
            authorizedRequest(path: "v1/routing/tasks/\(taskID)/cancel", method: "POST"),
            as: LoomRouteOperation.self
        )
        guard operation.schema == "loom-route-operation-v1",
              operation.decision.taskId == taskID,
              operation.receipt.taskId == taskID,
              operation.receipt.producingLanguage == "Sounio",
              operation.receipt.languageRole == "SEMANTIC_AUTHORITY"
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return operation
    }
}

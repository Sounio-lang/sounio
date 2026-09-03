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
        request.timeoutInterval = 4
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

    public func send(_ message: LoomMessageRequest) async throws -> LoomMessageReceipt {
        let url = baseURL.appending(path: "v1/messages")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 12
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(capability)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(message)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw LoomMessageClientError.invalidReceipt
        }
        guard http.statusCode == 202 else {
            let reason = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data).error)
                ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw LoomMessageClientError.refused(status: http.statusCode, reason: reason)
        }
        let receipt = try JSONDecoder().decode(LoomMessageReceipt.self, from: data)
        guard receipt.schema == "loom-message-receipt-v1",
              receipt.status == "accepted",
              receipt.messageId.isEmpty == false,
              receipt.threadId.isEmpty == false
        else {
            throw LoomMessageClientError.invalidReceipt
        }
        return receipt
    }
}

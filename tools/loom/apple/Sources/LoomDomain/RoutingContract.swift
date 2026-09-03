import Foundation

public enum QuotaState: String, Codable, CaseIterable, Hashable, Sendable {
    case exact
    case estimated
    case unknown
}

public enum PoolHealth: String, Codable, CaseIterable, Hashable, Sendable {
    case healthy
    case degraded
    case exhausted
    case authRequired = "auth_required"
}

public enum AdapterHealth: String, Codable, CaseIterable, Hashable, Sendable {
    case healthy
    case broken
    case missing
    case authRequired = "auth_required"
}

public enum ReceiptStatus: String, Codable, CaseIterable, Hashable, Sendable {
    case planned
    case running
    case committed
    case refused
    case fallback
}

public struct ProviderAccount: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let provider: String
    public let displayName: String

    public init(id: String, provider: String, displayName: String) {
        self.id = id
        self.provider = provider
        self.displayName = displayName
    }
}

public struct QuotaPool: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let accountId: String
    public let name: String
    public let state: QuotaState
    public let health: PoolHealth
    public let remainingFraction: Double?
    public let cooldownSeconds: Int?

    public init(
        id: String,
        accountId: String,
        name: String,
        state: QuotaState,
        health: PoolHealth,
        remainingFraction: Double?,
        cooldownSeconds: Int? = nil
    ) {
        self.id = id
        self.accountId = accountId
        self.name = name
        self.state = state
        self.health = health
        self.remainingFraction = remainingFraction
        self.cooldownSeconds = cooldownSeconds
    }
}

public struct CliAdapter: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let provider: String
    public let executable: String
    public let health: AdapterHealth
    public let version: String?

    public init(
        id: String,
        provider: String,
        executable: String,
        health: AdapterHealth,
        version: String? = nil
    ) {
        self.id = id
        self.provider = provider
        self.executable = executable
        self.health = health
        self.version = version
    }
}

public struct LoomTask: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let owner: String
    public let requiredCapabilities: [String]

    public init(id: String, title: String, owner: String, requiredCapabilities: [String]) {
        self.id = id
        self.title = title
        self.owner = owner
        self.requiredCapabilities = requiredCapabilities
    }
}

public struct RouteDecision: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let taskId: String
    public let policy: String
    public let candidateAdapterIds: [String]
    public let selectedAdapterId: String?

    public init(
        id: String,
        taskId: String,
        policy: String,
        candidateAdapterIds: [String],
        selectedAdapterId: String?
    ) {
        self.id = id
        self.taskId = taskId
        self.policy = policy
        self.candidateAdapterIds = candidateAdapterIds
        self.selectedAdapterId = selectedAdapterId
    }
}

public struct RouteReceipt: Identifiable, Codable, Equatable, Sendable {
    public var id: String { taskId }

    public let taskId: String
    public let policy: String
    public let poolId: String
    public let adapterId: String
    public let model: String
    public let effort: String
    public let reason: String
    public let fallbackChain: [String]
    public let status: ReceiptStatus

    public init(
        taskId: String,
        policy: String,
        poolId: String,
        adapterId: String,
        model: String,
        effort: String,
        reason: String,
        fallbackChain: [String],
        status: ReceiptStatus
    ) {
        self.taskId = taskId
        self.policy = policy
        self.poolId = poolId
        self.adapterId = adapterId
        self.model = model
        self.effort = effort
        self.reason = reason
        self.fallbackChain = fallbackChain
        self.status = status
    }
}

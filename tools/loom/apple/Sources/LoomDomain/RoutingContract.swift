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
    case completed
    case cancelled
    case committed
    case refused
    case fallback
    case failed
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
    public let sourceHash: String?
    public let semanticsHash: String?
    public let producingLanguage: String?
    public let languageRole: String?
    public let operationalLanguage: String?
    public let providerRole: String?
    public let toolchain: String?
    public let hardware: String?
    public let commandSha256: String?
    public let result: String?
    public let configHash: String?
    public let authorityOutputHash: String?
    public let providerPlanHash: String?
    public let quotaState: QuotaState?
    public let poolHealth: PoolHealth?
    public let adapterHealth: AdapterHealth?
    public let quotaUsedPercent: Double?
    public let quotaResetsAt: Int?
    public let quotaObservedUtc: String?
    public let quotaObservationHash: String?
    public let adapterObservationHash: String?
    public let sessionId: String?

    public init(
        taskId: String,
        policy: String,
        poolId: String,
        adapterId: String,
        model: String,
        effort: String,
        reason: String,
        fallbackChain: [String],
        status: ReceiptStatus,
        sourceHash: String? = nil,
        semanticsHash: String? = nil,
        producingLanguage: String? = nil,
        languageRole: String? = nil,
        operationalLanguage: String? = nil,
        providerRole: String? = nil,
        toolchain: String? = nil,
        hardware: String? = nil,
        commandSha256: String? = nil,
        result: String? = nil,
        configHash: String? = nil,
        authorityOutputHash: String? = nil,
        providerPlanHash: String? = nil,
        quotaState: QuotaState? = nil,
        poolHealth: PoolHealth? = nil,
        adapterHealth: AdapterHealth? = nil,
        quotaUsedPercent: Double? = nil,
        quotaResetsAt: Int? = nil,
        quotaObservedUtc: String? = nil,
        quotaObservationHash: String? = nil,
        adapterObservationHash: String? = nil,
        sessionId: String? = nil
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
        self.sourceHash = sourceHash
        self.semanticsHash = semanticsHash
        self.producingLanguage = producingLanguage
        self.languageRole = languageRole
        self.operationalLanguage = operationalLanguage
        self.providerRole = providerRole
        self.toolchain = toolchain
        self.hardware = hardware
        self.commandSha256 = commandSha256
        self.result = result
        self.configHash = configHash
        self.authorityOutputHash = authorityOutputHash
        self.providerPlanHash = providerPlanHash
        self.quotaState = quotaState
        self.poolHealth = poolHealth
        self.adapterHealth = adapterHealth
        self.quotaUsedPercent = quotaUsedPercent
        self.quotaResetsAt = quotaResetsAt
        self.quotaObservedUtc = quotaObservedUtc
        self.quotaObservationHash = quotaObservationHash
        self.adapterObservationHash = adapterObservationHash
        self.sessionId = sessionId
    }
}

public struct LoomRouteTaskRequest: Codable, Equatable, Sendable {
    public let schema: String
    public let taskId: String
    public let kind: String
    public let title: String
    public let prompt: String

    public init(taskId: String, title: String, prompt: String) {
        self.schema = "loom-route-task-v1"
        self.taskId = taskId
        self.kind = "review"
        self.title = title
        self.prompt = prompt
    }
}

public struct LoomRouteOperation: Codable, Equatable, Sendable {
    public let schema: String
    public let decision: RouteDecision
    public let receipt: RouteReceipt
}

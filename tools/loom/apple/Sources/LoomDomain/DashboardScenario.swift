import Foundation

public enum DashboardScenario: String, CaseIterable, Identifiable, Sendable {
    case nominal
    case estimatedQuota = "estimated_quota"
    case quotaExhausted = "quota_exhausted"
    case cliBroken = "cli_broken"
    case adapterMissing = "adapter_missing"
    case authExpired = "auth_expired"
    case cooldown
    case unknownQuota = "unknown_quota"
    case modelUnavailable = "model_unavailable"
    case ownershipBlock = "ownership_block"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .nominal: "Nominal"
        case .estimatedQuota: "Estimated quota"
        case .quotaExhausted: "Quota exhausted"
        case .cliBroken: "CLI broken"
        case .adapterMissing: "Adapter missing"
        case .authExpired: "Auth expired"
        case .cooldown: "Cooldown"
        case .unknownQuota: "Unknown quota"
        case .modelUnavailable: "Model unavailable"
        case .ownershipBlock: "Ownership block"
        }
    }
}

public struct AgentMessage: Identifiable, Equatable, Sendable {
    public let id: String
    public let author: String
    public let body: String
    public let isLocal: Bool

    public init(id: String, author: String, body: String, isLocal: Bool) {
        self.id = id
        self.author = author
        self.body = body
        self.isLocal = isLocal
    }
}

public struct DashboardSnapshot: Sendable {
    public let scenario: DashboardScenario
    public let accounts: [ProviderAccount]
    public let pools: [QuotaPool]
    public let adapters: [CliAdapter]
    public let task: LoomTask
    public let decision: RouteDecision
    public let receipt: RouteReceipt
    public let messages: [AgentMessage]

    public static func mock(_ scenario: DashboardScenario) -> DashboardSnapshot {
        let account = ProviderAccount(
            id: "account-openai-team",
            provider: "openai",
            displayName: "OpenAI Team"
        )
        let poolHealth: PoolHealth = switch scenario {
        case .quotaExhausted: .exhausted
        case .authExpired: .authRequired
        case .cooldown, .unknownQuota: .degraded
        default: .healthy
        }
        let quotaState: QuotaState = switch scenario {
        case .unknownQuota: .unknown
        case .estimatedQuota: .estimated
        default: .exact
        }
        let pool = QuotaPool(
            id: "pool-openai-team",
            accountId: account.id,
            name: "Team pool",
            state: quotaState,
            health: poolHealth,
            remainingFraction: quotaState == .unknown ? nil : (scenario == .quotaExhausted ? 0 : 0.72),
            cooldownSeconds: scenario == .cooldown ? 412 : nil
        )
        let adapterHealth: AdapterHealth = switch scenario {
        case .cliBroken: .broken
        case .adapterMissing: .missing
        case .authExpired: .authRequired
        default: .healthy
        }
        let adapter = CliAdapter(
            id: "adapter-codex",
            provider: "openai",
            executable: "codex",
            health: adapterHealth,
            version: "provider-native"
        )
        let task = LoomTask(
            id: "task-loom-ui-41",
            title: "Validate native Loom observatory",
            owner: scenario == .ownershipBlock ? "claude-2" : "codex-1",
            requiredCapabilities: ["swiftui", "metal", "loom-read"]
        )
        let refused = switch scenario {
        case .nominal, .estimatedQuota: false
        default: true
        }
        let reason: String = switch scenario {
        case .nominal: "Policy selected the healthy exact-quota path."
        case .estimatedQuota: "The pool exposes an estimate, not an exact quota reading."
        case .quotaExhausted: "The selected quota pool is exhausted."
        case .cliBroken: "The provider CLI adapter failed its native probe."
        case .adapterMissing: "The provider CLI executable is absent on this host."
        case .authExpired: "Authentication requires provider-native renewal."
        case .cooldown: "The pool is healthy enough to recover after cooldown."
        case .unknownQuota: "Quota truth is unavailable; the UI cannot infer capacity."
        case .modelUnavailable: "The requested model is not exposed by this account."
        case .ownershipBlock: "Another lane owns the requested write surface."
        }
        let receipt = RouteReceipt(
            taskId: task.id,
            policy: "authority-first",
            poolId: pool.id,
            adapterId: adapter.id,
            model: scenario == .modelUnavailable ? "gpt-5.6-sol-unavailable" : "gpt-5.6-terra",
            effort: "high",
            reason: reason,
            fallbackChain: refused ? ["gpt-5.6-terra", "gpt-5.6-sol"] : [],
            status: refused ? .refused : .committed
        )
        let decision = RouteDecision(
            id: "decision-" + task.id,
            taskId: task.id,
            policy: receipt.policy,
            candidateAdapterIds: [adapter.id, "adapter-claude"],
            selectedAdapterId: refused ? nil : adapter.id
        )
        return DashboardSnapshot(
            scenario: scenario,
            accounts: [account],
            pools: [pool],
            adapters: [adapter],
            task: task,
            decision: decision,
            receipt: receipt,
            messages: [
                AgentMessage(
                    id: "m1",
                    author: "codex-3",
                    body: "Topology receipt is ready. The selected filament preserves backend authority.",
                    isLocal: false
                ),
                AgentMessage(
                    id: "m2",
                    author: "you",
                    body: "Keep the session here. Bring forward skills, context, and evidence.",
                    isLocal: true
                ),
            ]
        )
    }
}

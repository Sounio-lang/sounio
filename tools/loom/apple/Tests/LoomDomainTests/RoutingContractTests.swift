import XCTest
@testable import LoomDomain

final class RoutingContractTests: XCTestCase {
    func testAuthorityEnumsHaveExactWireValues() {
        XCTAssertEqual(QuotaState.allCases.map(\.rawValue), ["exact", "estimated", "unknown"])
        XCTAssertEqual(
            PoolHealth.allCases.map(\.rawValue),
            ["healthy", "degraded", "exhausted", "auth_required"]
        )
        XCTAssertEqual(
            AdapterHealth.allCases.map(\.rawValue),
            ["healthy", "broken", "missing", "auth_required"]
        )
    }

    func testEveryRequiredNegativeStateHasAMock() {
        let states = Set(DashboardScenario.allCases.map(\.rawValue))
        XCTAssertEqual(
            states,
            Set([
                "nominal",
                "estimated_quota",
                "quota_exhausted",
                "cli_broken",
                "adapter_missing",
                "auth_expired",
                "cooldown",
                "unknown_quota",
                "model_unavailable",
                "ownership_block",
            ])
        )
        for scenario in DashboardScenario.allCases {
            XCTAssertEqual(DashboardSnapshot.mock(scenario).scenario, scenario)
        }
    }

    func testMocksCoverEveryQuotaPoolAndAdapterWireState() {
        let mocks = DashboardScenario.allCases.map(DashboardSnapshot.mock)
        XCTAssertEqual(Set(mocks.map { $0.pools[0].state }), Set(QuotaState.allCases))
        XCTAssertEqual(Set(mocks.map { $0.pools[0].health }), Set(PoolHealth.allCases))
        XCTAssertEqual(Set(mocks.map { $0.adapters[0].health }), Set(AdapterHealth.allCases))
    }

    func testRouteReceiptRoundTripsWithoutAuthorityPromotion() throws {
        let receipt = DashboardSnapshot.mock(.nominal).receipt
        let data = try JSONEncoder().encode(receipt)
        let decoded = try JSONDecoder().decode(RouteReceipt.self, from: data)
        XCTAssertEqual(decoded, receipt)
        XCTAssertEqual(decoded.status, .committed)
    }

    func testProviderAccountPoolAndAdapterRemainSeparate() {
        let snapshot = DashboardSnapshot.mock(.nominal)
        XCTAssertNotEqual(snapshot.accounts[0].id, snapshot.pools[0].id)
        XCTAssertNotEqual(snapshot.pools[0].id, snapshot.adapters[0].id)
        XCTAssertEqual(snapshot.pools[0].accountId, snapshot.accounts[0].id)
    }

    func testPreferredDeliveryLaneRequiresLivePresenceAndActiveEndpointFirst() {
        let liveStale = lane("live-stale", presence: "live", endpoint: "stale")
        let absentActive = lane("absent-active", presence: "missing", endpoint: "active")
        let liveActive = lane("live-active", presence: "live", endpoint: "active")

        XCTAssertEqual(
            LoomFleetSnapshot.Lane.preferredDeliveryLane(
                in: [liveStale, absentActive, liveActive]
            ),
            liveActive
        )
        XCTAssertEqual(liveActive.deliveryReadiness, .immediate)
        XCTAssertEqual(liveStale.deliveryReadiness, .durableOnly)
    }

    func testPreferredDeliveryLaneStillAllowsDurableOnlyFallback() {
        let missing = lane("missing", presence: "missing", endpoint: "missing")
        let liveStale = lane("live-stale", presence: "live", endpoint: "stale")

        XCTAssertEqual(
            LoomFleetSnapshot.Lane.preferredDeliveryLane(in: [missing, liveStale]),
            liveStale
        )
    }

    func testThreadTruthWireContractKeepsBusEventsDistinct() throws {
        let data = Data("""
        {
          "schema":"loom-message-thread-v1",
          "request":{"id":"msg-1","utc":"2026-09-03T00:00:00Z","createdEpoch":1,"fromAgent":"founder-ui","fromLane":"loom-apple","toAgent":"claude","toLane":"lane","kind":"request","text":"hello","threadId":"msg-1","replyTo":""},
          "state":"answered",
          "delivery":"wake_received",
          "injected":1,
          "acknowledged":0,
          "responseCount":1,
          "wakeCount":1,
          "wakePending":0,
          "timeoutSeconds":60,
          "events":[
            {"id":"request:msg-1","utc":"2026-09-03T00:00:00Z","kind":"request","state":"accepted","messageId":"msg-1","actor":"founder-ui","body":"hello"},
            {"id":"wake:msg-1","utc":"2026-09-03T00:00:01Z","kind":"wake","state":"received","messageId":"msg-1","actor":"loom-delivery","body":"wake"},
            {"id":"response:msg-2","utc":"2026-09-03T00:00:02Z","kind":"response","state":"reply","messageId":"msg-2","actor":"claude/lane","body":"received"},
            {"id":"ack:msg-2","utc":"2026-09-03T00:00:03Z","kind":"ack","state":"acknowledged","messageId":"msg-2","actor":"founder-ui","body":"ack"}
          ]
        }
        """.utf8)
        let thread = try JSONDecoder().decode(LoomMessageThread.self, from: data)

        XCTAssertEqual(thread.schema, "loom-message-thread-v1")
        XCTAssertEqual(thread.state, "answered")
        XCTAssertEqual(thread.delivery, "wake_received")
        XCTAssertEqual(thread.events.map(\.kind), ["request", "wake", "response", "ack"])
        XCTAssertEqual(thread.events.map(\.isLocal), [true, false, false, true])
    }

    func testThreadTruthSupportsTimeoutAndDurableOnlyStates() throws {
        let data = Data("""
        {"schema":"loom-message-thread-v1","request":{"id":"msg-timeout","utc":"2026-09-03T00:00:00Z","createdEpoch":1,"fromAgent":"founder-ui","fromLane":"loom-apple","toAgent":"grok","toLane":"fleet","kind":"request","text":"wait","threadId":"msg-timeout","replyTo":""},"state":"timed_out","delivery":"durable_only","injected":0,"acknowledged":0,"responseCount":0,"wakeCount":0,"wakePending":0,"timeoutSeconds":60,"events":[{"id":"durable:msg-timeout","utc":"2026-09-03T00:00:00Z","kind":"durable_only","state":"stored","messageId":"msg-timeout","actor":"loom-bus","body":"stored"},{"id":"timeout:msg-timeout","utc":"2026-09-03T00:01:00Z","kind":"timeout","state":"elapsed","messageId":"msg-timeout","actor":"loom-clock","body":"elapsed"}]}
        """.utf8)
        let thread = try JSONDecoder().decode(LoomMessageThread.self, from: data)

        XCTAssertEqual(thread.state, "timed_out")
        XCTAssertEqual(thread.delivery, "durable_only")
        XCTAssertEqual(thread.events.map(\.kind), ["durable_only", "timeout"])
    }

    private func lane(
        _ lane: String,
        presence: String,
        endpoint: String
    ) -> LoomFleetSnapshot.Lane {
        LoomFleetSnapshot.Lane(
            agent: "test",
            lane: lane,
            state: presence,
            claimState: "none",
            presenceState: presence,
            endpointState: endpoint,
            harness: "test",
            worktree: "/tmp/test",
            loomState: "none",
            cursor: 0
        )
    }
}

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

import SwiftUI
import LoomDomain

struct TopologyWeaveView: View {
    let snapshot: DashboardSnapshot
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var hoverUnit = CGPoint(x: 0.5, y: 0.5)

    var body: some View {
        TimelineView(.animation(minimumInterval: reduceMotion ? 1 : 1 / 30)) { timeline in
            GeometryReader { proxy in
                let phase = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate
                let points = routePoints(in: proxy.size)

                ZStack {
                    Canvas { context, size in
                        drawSpatialStage(context: &context, size: size, phase: phase)
                    }
                    .allowsHitTesting(false)

                    stratumLabel(number: "01", title: "QUOTA POOLS", color: LoomColor.green)
                        .position(x: 88, y: proxy.size.height * 0.135)
                    stratumLabel(number: "02", title: "CLI ADAPTERS", color: LoomColor.cyan)
                        .position(x: 88, y: proxy.size.height * 0.415)
                    stratumLabel(number: "03", title: "MODELS", color: LoomColor.magenta)
                        .position(x: 88, y: proxy.size.height * 0.695)

                    topologyPlate(
                        title: "QUOTA POOL",
                        detail: snapshot.pools[0].name,
                        value: snapshot.pools[0].health.rawValue,
                        icon: "gauge.with.dots.needle.67percent",
                        color: snapshot.pools[0].health.loomColor,
                        depth: 0
                    )
                    .position(points[0])

                    topologyPlate(
                        title: "CLI ADAPTER",
                        detail: snapshot.adapters[0].executable,
                        value: snapshot.adapters[0].health.rawValue,
                        icon: "terminal",
                        color: snapshot.adapters[0].health.loomColor,
                        depth: 1
                    )
                    .position(points[1])

                    topologyPlate(
                        title: "MODEL",
                        detail: snapshot.receipt.model,
                        value: snapshot.receipt.effort,
                        icon: "cpu",
                        color: snapshot.receipt.status.loomColor,
                        depth: 2
                    )
                    .position(points[2])

                    routeNode(color: snapshot.receipt.status.loomColor, phase: phase)
                        .position(points[1])
                }
                .contentShape(Rectangle())
                .onContinuousHover { hoverPhase in
                    guard !reduceMotion else { return }
                    switch hoverPhase {
                    case .active(let location):
                        hoverUnit = CGPoint(
                            x: min(max(location.x / max(proxy.size.width, 1), 0), 1),
                            y: min(max(location.y / max(proxy.size.height, 1), 0), 1)
                        )
                    case .ended:
                        hoverUnit = CGPoint(x: 0.5, y: 0.5)
                    }
                }
                .animation(.spring(response: 0.32, dampingFraction: 0.82), value: hoverUnit)
            }
        }
    }

    private func stratumLabel(number: String, title: String, color: Color) -> some View {
        HStack(spacing: 7) {
            Text(number)
                .foregroundStyle(color)
            Text(title)
                .foregroundStyle(.secondary)
        }
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .frame(width: 150, alignment: .leading)
    }

    private func topologyPlate(
        title: String,
        detail: String,
        value: String,
        icon: String,
        color: Color,
        depth: Int
    ) -> some View {
        GlassSurface {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 7) {
                    Image(systemName: icon)
                        .foregroundStyle(color)
                    Text(title)
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 4)
                    Circle()
                        .fill(color)
                        .frame(width: 6, height: 6)
                        .shadow(color: color, radius: 6)
                }
                Text(detail)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                Text(value.uppercased())
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(color)
                    .lineLimit(1)
            }
            .padding(12)
            .frame(width: 192, height: 92)
        }
        .rotation3DEffect(
            .degrees(7 + (hoverUnit.y - 0.5) * 5),
            axis: (x: 1, y: 0, z: 0),
            perspective: 0.42
        )
        .rotation3DEffect(
            .degrees((hoverUnit.x - 0.5) * -7),
            axis: (x: 0, y: 1, z: 0),
            perspective: 0.42
        )
        .offset(x: (hoverUnit.x - 0.5) * CGFloat(4 + depth * 3))
        .shadow(color: color.opacity(0.18), radius: 26, y: 14)
    }

    private func routeNode(color: Color, phase: Double) -> some View {
        let pulse = 1 + 0.12 * sin(phase * 2.4)
        return ZStack {
            Circle()
                .stroke(color.opacity(0.32), lineWidth: 1)
                .frame(width: 52, height: 52)
                .scaleEffect(pulse)
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
                .shadow(color: color, radius: 12)
        }
        .accessibilityHidden(true)
    }

    private func routePoints(in size: CGSize) -> [CGPoint] {
        [
            CGPoint(x: size.width * 0.34, y: size.height * 0.22),
            CGPoint(x: size.width * 0.52, y: size.height * 0.50),
            CGPoint(x: size.width * 0.70, y: size.height * 0.78),
        ]
    }

    private func drawSpatialStage(
        context: inout GraphicsContext,
        size: CGSize,
        phase: Double
    ) {
        let planeColors = [LoomColor.green, LoomColor.cyan, LoomColor.magenta]
        for index in 0..<3 {
            drawPlane(context: &context, size: size, index: index, color: planeColors[index])
        }
        drawFilaments(context: &context, size: size, phase: phase)
    }

    private func drawPlane(
        context: inout GraphicsContext,
        size: CGSize,
        index: Int,
        color: Color
    ) {
        let centerY = size.height * [0.22, 0.50, 0.78][index]
        let halfHeight = size.height * 0.105
        let corners = [
            CGPoint(x: size.width * 0.045, y: centerY - halfHeight + 7),
            CGPoint(x: size.width * 0.90, y: centerY - halfHeight),
            CGPoint(x: size.width * 0.96, y: centerY + halfHeight),
            CGPoint(x: size.width * 0.105, y: centerY + halfHeight + 7),
        ]

        var plane = Path()
        plane.move(to: corners[0])
        for corner in corners.dropFirst() { plane.addLine(to: corner) }
        plane.closeSubpath()

        let trailingColor = index == 2 ? LoomColor.magenta : LoomColor.cyan
        context.fill(
            plane,
            with: .linearGradient(
                Gradient(colors: [
                    color.opacity(reduceTransparency ? 0.06 : 0.12),
                    Color.black.opacity(0.025),
                    trailingColor.opacity(reduceTransparency ? 0.025 : 0.07),
                ]),
                startPoint: corners[0],
                endPoint: corners[2]
            )
        )
        context.stroke(plane, with: .color(color.opacity(0.34)), lineWidth: 0.8)

        for step in 1..<7 {
            let amount = CGFloat(step) / 7
            var depthLine = Path()
            depthLine.move(to: interpolate(corners[0], corners[3], amount))
            depthLine.addLine(to: interpolate(corners[1], corners[2], amount))
            context.stroke(depthLine, with: .color(color.opacity(0.075)), lineWidth: 0.55)
        }

        for step in 1..<9 {
            let amount = CGFloat(step) / 9
            var ray = Path()
            ray.move(to: interpolate(corners[0], corners[1], amount))
            ray.addLine(to: interpolate(corners[3], corners[2], amount))
            context.stroke(ray, with: .color(color.opacity(0.055)), lineWidth: 0.5)
        }
    }

    private func drawFilaments(context: inout GraphicsContext, size: CGSize, phase: Double) {
        let points = routePoints(in: size)
        var path = Path()
        path.move(to: points[0])
        path.addCurve(
            to: points[1],
            control1: CGPoint(x: size.width * 0.43, y: size.height * 0.25),
            control2: CGPoint(x: size.width * 0.43, y: size.height * 0.47)
        )
        path.addCurve(
            to: points[2],
            control1: CGPoint(x: size.width * 0.61, y: size.height * 0.53),
            control2: CGPoint(x: size.width * 0.61, y: size.height * 0.75)
        )

        let color = snapshot.receipt.status.loomColor
        context.drawLayer { glow in
            glow.addFilter(.blur(radius: 14))
            glow.stroke(path, with: .color(color.opacity(0.46)), lineWidth: 11)
        }
        context.stroke(
            path,
            with: .linearGradient(
                Gradient(colors: [LoomColor.green, LoomColor.cyan, color]),
                startPoint: points[0],
                endPoint: points[2]
            ),
            style: StrokeStyle(lineWidth: 3.2, lineCap: .round)
        )
        context.stroke(
            path,
            with: .color(Color.white.opacity(0.42)),
            style: StrokeStyle(
                lineWidth: 0.8,
                lineCap: .round,
                dash: [1, 9],
                dashPhase: phase * -32
            )
        )

        let progress = (phase * 0.16).truncatingRemainder(dividingBy: 1)
        let segment = progress < 0.5 ? 0 : 1
        let amount = CGFloat((progress - Double(segment) * 0.5) * 2)
        let traveler = interpolate(points[segment], points[segment + 1], amount)
        context.drawLayer { glow in
            glow.addFilter(.blur(radius: 9))
            glow.fill(
                Path(ellipseIn: CGRect(x: traveler.x - 8, y: traveler.y - 8, width: 16, height: 16)),
                with: .color(color.opacity(0.9))
            )
        }
        context.fill(
            Path(ellipseIn: CGRect(x: traveler.x - 3, y: traveler.y - 3, width: 6, height: 6)),
            with: .color(.white)
        )
    }

    private func interpolate(_ start: CGPoint, _ end: CGPoint, _ amount: CGFloat) -> CGPoint {
        CGPoint(
            x: start.x + (end.x - start.x) * amount,
            y: start.y + (end.y - start.y) * amount
        )
    }
}

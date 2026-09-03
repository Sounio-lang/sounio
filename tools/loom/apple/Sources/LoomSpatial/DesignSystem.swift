import SwiftUI
import LoomDomain

enum LoomColor {
    static let ink = Color(red: 0.035, green: 0.045, blue: 0.060)
    static let graphite = Color(red: 0.075, green: 0.085, blue: 0.105)
    static let cyan = Color(red: 0.26, green: 0.86, blue: 0.94)
    static let magenta = Color(red: 0.92, green: 0.35, blue: 0.78)
    static let amber = Color(red: 0.98, green: 0.68, blue: 0.22)
    static let green = Color(red: 0.34, green: 0.90, blue: 0.56)
    static let red = Color(red: 1.00, green: 0.38, blue: 0.38)
    static let violet = Color(red: 0.52, green: 0.46, blue: 0.96)
}

struct GlassSurface<Content: View>: View {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast

    let radius: CGFloat
    let content: Content

    init(radius: CGFloat = 8, @ViewBuilder content: () -> Content) {
        self.radius = radius
        self.content = content()
    }

    var body: some View {
        content
            .background {
                ZStack {
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(reduceTransparency ? LoomColor.graphite : Color.black.opacity(0.36))
                    if !reduceTransparency {
                        RoundedRectangle(cornerRadius: radius, style: .continuous)
                            .fill(.ultraThinMaterial)
                        RoundedRectangle(cornerRadius: radius, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [
                                        Color.white.opacity(0.10),
                                        LoomColor.cyan.opacity(0.025),
                                        LoomColor.magenta.opacity(0.018),
                                        Color.clear,
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                    }
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(contrast == .increased ? 0.48 : 0.28),
                                LoomColor.cyan.opacity(0.16),
                                Color.white.opacity(0.06),
                                LoomColor.magenta.opacity(0.10),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: contrast == .increased ? 1.4 : 0.8
                    )
            }
            .overlay(alignment: .top) {
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [Color.clear, Color.white.opacity(0.32), Color.clear],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(height: 1)
                    .padding(.horizontal, 12)
                    .opacity(reduceTransparency ? 0 : 1)
            }
            .shadow(color: .black.opacity(0.34), radius: 30, y: 16)
            .shadow(color: LoomColor.cyan.opacity(0.035), radius: 18, y: -2)
            .compositingGroup()
    }
}

struct StatusPill: View {
    let label: String
    let color: Color
    var systemImage: String? = nil

    var body: some View {
        HStack(spacing: 6) {
            if let systemImage {
                Image(systemName: systemImage)
            } else {
                Circle().fill(color).frame(width: 6, height: 6)
            }
            Text(label.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
        }
        .foregroundStyle(color)
        .lineLimit(1)
        .padding(.horizontal, 8)
        .frame(height: 24)
        .background(color.opacity(0.10), in: Capsule())
        .overlay(Capsule().stroke(color.opacity(0.26), lineWidth: 0.8))
    }
}

extension PoolHealth {
    var loomColor: Color {
        switch self {
        case .healthy: LoomColor.green
        case .degraded: LoomColor.amber
        case .exhausted, .authRequired: LoomColor.red
        }
    }
}

extension AdapterHealth {
    var loomColor: Color {
        switch self {
        case .healthy: LoomColor.green
        case .broken, .missing, .authRequired: LoomColor.red
        }
    }
}

extension ReceiptStatus {
    var loomColor: Color {
        switch self {
        case .committed, .completed: LoomColor.cyan
        case .running: LoomColor.green
        case .fallback: LoomColor.amber
        case .planned: LoomColor.violet
        case .cancelled: LoomColor.amber
        case .refused, .failed: LoomColor.red
        }
    }
}

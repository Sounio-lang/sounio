// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SounioSite",
    platforms: [.macOS(.v13)],
    targets: [
        // A camada de evidência é um módulo separado de propósito: é o limite
        // de módulo que faz `Claim.init` ser inalcançável de fora.
        .target(name: "Evidence"),
        .executableTarget(name: "sitegen", dependencies: ["Evidence"]),
    ]
)

// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "LoomApple",
    platforms: [
        .macOS("27.0"),
        .iOS("27.0"),
    ],
    products: [
        .library(name: "LoomDomain", targets: ["LoomDomain"]),
        .executable(name: "LoomSpatial", targets: ["LoomSpatial"]),
    ],
    targets: [
        .target(name: "LoomDomain"),
        .executableTarget(
            name: "LoomSpatial",
            dependencies: ["LoomDomain"],
            resources: [.process("Resources")]
        ),
        .testTarget(name: "LoomDomainTests", dependencies: ["LoomDomain"]),
    ]
)

// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "kernel-auto-tuner",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "MetalHello", targets: ["MetalHello"]),
    ],
    targets: [
        .executableTarget(
            name: "MetalHello",
            resources: [
                .process("Resources"),
            ]
        ),
    ]
)

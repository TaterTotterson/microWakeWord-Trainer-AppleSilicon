// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "WakeWordTrainer",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "WakeWordTrainer", targets: ["WakeWordTrainer"])
    ],
    targets: [
        .executableTarget(
            name: "WakeWordTrainer",
            path: "Sources/WakeWordTrainer",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("WebKit")
            ]
        )
    ]
)

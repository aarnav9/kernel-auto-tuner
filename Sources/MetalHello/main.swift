import Foundation

struct TuningTarget {
    let name: String
    let category: String
    let shape: MatMulShape
}

struct SweepRecord {
    let target: TuningTarget
    let kernel: String
    let config: TiledKernelConfig?
    let result: BenchmarkResult
    let tuningScore: Double?
}

struct ExhaustiveBestRecord {
    let target: TuningTarget
    let evaluation: SAEvaluation
}

struct FusedSweepRecord {
    let target: TuningTarget
    let kernel: String
    let config: TiledKernelConfig?
    let result: BenchmarkResult
    let tuningScore: Double?
}

struct ShapeHuntRecord {
    let target: TuningTarget
    let bestEvaluation: SAEvaluation
    let mpsResult: BenchmarkResult
    let bestConfidence: ConfidenceSummary
    let mpsConfidence: ConfidenceSummary

    var confirmedRatio: Double {
        bestConfidence.medianOfMedians / mpsConfidence.medianOfMedians
    }

    var confirmedWin: Bool {
        confirmedRatio < 1.0
    }
}

let vectorLengths = [16, 1 << 16]
let correctnessShapes = [
    MatMulShape(m: 32, n: 32, k: 32),
    MatMulShape(m: 64, n: 96, k: 80),
    MatMulShape(m: 65, n: 97, k: 33),
    MatMulShape(m: 17, n: 19, k: 21),
    MatMulShape(m: 31, n: 47, k: 15),
    MatMulShape(m: 130, n: 130, k: 130),
]
let tuningTargets = [
    TuningTarget(name: "square128", category: "square", shape: MatMulShape(m: 128, n: 128, k: 128)),
    TuningTarget(name: "square256", category: "square", shape: MatMulShape(m: 256, n: 256, k: 256)),
    TuningTarget(name: "medium_rect", category: "medium_rect", shape: MatMulShape(m: 192, n: 320, k: 96)),
    TuningTarget(name: "edge_square", category: "edge_heavy", shape: MatMulShape(m: 130, n: 130, k: 130)),
    TuningTarget(name: "edge_rect", category: "irregular", shape: MatMulShape(m: 255, n: 257, k: 129)),
]
let fusedTargets = [
    TuningTarget(name: "fused_square128", category: "fused_square", shape: MatMulShape(m: 128, n: 128, k: 128)),
    TuningTarget(name: "fused_medium_rect", category: "fused_rect", shape: MatMulShape(m: 192, n: 320, k: 96)),
    TuningTarget(name: "fused_edge_rect", category: "fused_irregular", shape: MatMulShape(m: 255, n: 257, k: 129)),
]
let shapeHuntTargets = [
    TuningTarget(name: "hunt_square112", category: "shape_hunt", shape: MatMulShape(m: 112, n: 112, k: 112)),
    TuningTarget(name: "hunt_square144", category: "shape_hunt", shape: MatMulShape(m: 144, n: 144, k: 144)),
    TuningTarget(name: "hunt_edge129", category: "shape_hunt", shape: MatMulShape(m: 129, n: 129, k: 129)),
    TuningTarget(name: "hunt_rect160x96x64", category: "shape_hunt", shape: MatMulShape(m: 160, n: 96, k: 64)),
    TuningTarget(name: "hunt_rect96x160x64", category: "shape_hunt", shape: MatMulShape(m: 96, n: 160, k: 64)),
    TuningTarget(name: "hunt_irregular191x255x65", category: "shape_hunt", shape: MatMulShape(m: 191, n: 255, k: 65)),
]

let benchmarkIterations = 10
let benchmarkWarmup = 2
let stabilityThreshold = 1.5
let saIterations = 12
let saSeedBase: UInt64 = 0x5A17_2026
let benchmarkSeedBase: UInt64 = 0xBEEF_1000
let fusedBenchmarkSeedBase: UInt64 = 0xBEEF_2000
let confidencePassCount = 3
let confidenceIterations = 20
let confidenceWarmup = 3
let winValidationPassCount = 5
let winValidationIterations = 30
let winValidationWarmup = 5
let winValidationTopK = 3
let winValidationTargetNames = ["edge_square", "hunt_edge129", "hunt_rect160x96x64"]

func gflops(for shape: MatMulShape, milliseconds: Double) -> Double {
    let flops = 2.0 * Double(shape.m * shape.n * shape.k)
    return flops / (milliseconds / 1_000.0) / 1_000_000_000.0
}

func printBenchmark(label: String, shape: MatMulShape, result: BenchmarkResult) {
    let samples = result.samples.map { String(format: "%.3f", $0) }.joined(separator: ", ")
    let avgThroughput = gflops(for: shape, milliseconds: result.averageMilliseconds)
    let medianThroughput = gflops(for: shape, milliseconds: result.medianMilliseconds)
    print("\(label): avg=\(String(format: "%.3f", result.averageMilliseconds)) ms | median=\(String(format: "%.3f", result.medianMilliseconds)) ms | min=\(String(format: "%.3f", result.minMilliseconds)) ms | max=\(String(format: "%.3f", result.maxMilliseconds)) ms | stability=\(String(format: "%.2f", result.stabilityRatio)) | avg_gflops=\(String(format: "%.2f", avgThroughput)) | median_gflops=\(String(format: "%.2f", medianThroughput)) | samples=[\(samples)]")
}

func csvField(_ value: String) -> String {
    if value.contains(",") || value.contains("\"") || value.contains("\n") {
        return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
    }
    return value
}

func writeSweepResults(_ records: [SweepRecord]) throws {
    let resultsDir = try ensureResultsDirectory()

    let header = [
        "target",
        "category",
        "shape",
        "kernel",
        "family",
        "config",
        "tile_m",
        "tile_n",
        "tile_k",
        "threads_x",
        "threads_y",
        "average_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "stability_ratio",
        "stable",
        "tuning_score_ms",
        "average_gflops",
        "median_gflops",
    ]

    var lines = [header.joined(separator: ",")]
    for record in records {
        let cfg = record.config
        let configDescription = cfg?.description ?? ""
        let tileM = cfg.map { String($0.tileM) } ?? ""
        let tileN = cfg.map { String($0.tileN) } ?? ""
        let tileK = cfg.map { String($0.tileK) } ?? ""
        let threadsX = cfg.map { String($0.threadsX) } ?? ""
        let threadsY = cfg.map { String($0.threadsY) } ?? ""
        let averageMs = String(format: "%.6f", record.result.averageMilliseconds)
        let medianMs = String(format: "%.6f", record.result.medianMilliseconds)
        let minMs = String(format: "%.6f", record.result.minMilliseconds)
        let maxMs = String(format: "%.6f", record.result.maxMilliseconds)
        let stability = String(format: "%.6f", record.result.stabilityRatio)
        let stable = record.result.isStable(stabilityThreshold: stabilityThreshold) ? "true" : "false"
        let tuningScore = record.tuningScore.map { String(format: "%.6f", $0) } ?? ""
        let averageGFLOPS = String(format: "%.6f", gflops(for: record.target.shape, milliseconds: record.result.averageMilliseconds))
        let medianGFLOPS = String(format: "%.6f", gflops(for: record.target.shape, milliseconds: record.result.medianMilliseconds))

        let fields: [String] = [
            record.target.name,
            record.target.category,
            record.target.shape.description,
            record.kernel,
            record.config?.family.rawValue ?? "",
            configDescription,
            tileM,
            tileN,
            tileK,
            threadsX,
            threadsY,
            averageMs,
            medianMs,
            minMs,
            maxMs,
            stability,
            stable,
            tuningScore,
            averageGFLOPS,
            medianGFLOPS,
        ]
        lines.append(fields.map(csvField).joined(separator: ","))
    }

    try lines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("tuning_sweep.csv"),
        atomically: true,
        encoding: .utf8
    )
}

func ensureResultsDirectory() throws -> URL {
    let resultsDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("results", isDirectory: true)
    try FileManager.default.createDirectory(at: resultsDir, withIntermediateDirectories: true)
    return resultsDir
}

func writeSAResults(_ summaries: [SASummary]) throws {
    let resultsDir = try ensureResultsDirectory()

    let summaryHeader = [
        "target",
        "category",
        "shape",
        "seed",
        "start_config",
        "best_config",
        "best_median_ms",
        "best_tuning_score_ms",
        "best_stability_ratio",
        "exhaustive_best_config",
        "exhaustive_best_score_ms",
        "score_gap_ms",
        "regret_percent",
        "best_rank",
        "config_count",
        "matched_exhaustive_best",
        "evaluation_count",
    ]

    var summaryLines = [summaryHeader.joined(separator: ",")]
    for summary in summaries {
        let fields = [
            summary.target.name,
            summary.target.category,
            summary.target.shape.description,
            String(summary.seed),
            summary.startConfig.description,
            summary.bestEvaluation.config.description,
            String(format: "%.6f", summary.bestEvaluation.result.medianMilliseconds),
            String(format: "%.6f", summary.bestEvaluation.score),
            String(format: "%.6f", summary.bestEvaluation.result.stabilityRatio),
            summary.exhaustiveBest.config.description,
            String(format: "%.6f", summary.exhaustiveBest.score),
            String(format: "%.6f", summary.scoreGapMilliseconds),
            String(format: "%.6f", summary.regretPercent),
            String(summary.bestRank),
            String(summary.configCount),
            summary.matchedExhaustiveBest ? "true" : "false",
            String(summary.evaluationCount),
        ]
        summaryLines.append(fields.map(csvField).joined(separator: ","))
    }

    try summaryLines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("sa_summary.csv"),
        atomically: true,
        encoding: .utf8
    )

    let historyHeader = [
        "target",
        "category",
        "shape",
        "iteration",
        "temperature",
        "current_config",
        "candidate_config",
        "accepted",
        "current_score_ms",
        "best_config",
        "best_score_ms",
    ]

    var historyLines = [historyHeader.joined(separator: ",")]
    for summary in summaries {
        for entry in summary.history {
            let fields = [
                summary.target.name,
                summary.target.category,
                summary.target.shape.description,
                String(entry.iteration),
                String(format: "%.6f", entry.temperature),
                entry.currentConfig.description,
                entry.candidateConfig.description,
                entry.accepted ? "true" : "false",
                String(format: "%.6f", entry.currentScore),
                entry.bestConfig.description,
                String(format: "%.6f", entry.bestScore),
            ]
            historyLines.append(fields.map(csvField).joined(separator: ","))
        }
    }

    try historyLines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("sa_history.csv"),
        atomically: true,
        encoding: .utf8
    )
}

func writeFusedSweepResults(_ records: [FusedSweepRecord]) throws {
    let resultsDir = try ensureResultsDirectory()

    let header = [
        "target",
        "category",
        "shape",
        "kernel",
        "family",
        "config",
        "tile_m",
        "tile_n",
        "tile_k",
        "threads_x",
        "threads_y",
        "average_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "stability_ratio",
        "stable",
        "tuning_score_ms",
        "average_gflops",
        "median_gflops",
    ]

    var lines = [header.joined(separator: ",")]
    for record in records {
        let cfg = record.config
        let fields: [String] = [
            record.target.name,
            record.target.category,
            record.target.shape.description,
            record.kernel,
            cfg?.family.rawValue ?? "",
            cfg?.description ?? "",
            cfg.map { String($0.tileM) } ?? "",
            cfg.map { String($0.tileN) } ?? "",
            cfg.map { String($0.tileK) } ?? "",
            cfg.map { String($0.threadsX) } ?? "",
            cfg.map { String($0.threadsY) } ?? "",
            String(format: "%.6f", record.result.averageMilliseconds),
            String(format: "%.6f", record.result.medianMilliseconds),
            String(format: "%.6f", record.result.minMilliseconds),
            String(format: "%.6f", record.result.maxMilliseconds),
            String(format: "%.6f", record.result.stabilityRatio),
            record.result.isStable(stabilityThreshold: stabilityThreshold) ? "true" : "false",
            record.tuningScore.map { String(format: "%.6f", $0) } ?? "",
            String(format: "%.6f", gflops(for: record.target.shape, milliseconds: record.result.averageMilliseconds)),
            String(format: "%.6f", gflops(for: record.target.shape, milliseconds: record.result.medianMilliseconds)),
        ]
        lines.append(fields.map(csvField).joined(separator: ","))
    }

    try lines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("fused_tuning_sweep.csv"),
        atomically: true,
        encoding: .utf8
    )
}

func writeShapeHuntResults(_ records: [ShapeHuntRecord]) throws {
    let resultsDir = try ensureResultsDirectory()
    let header = [
        "target",
        "shape",
        "best_config",
        "best_median_ms",
        "mps_median_ms",
        "sweep_ratio_vs_mps",
        "confidence_custom_ms",
        "confidence_mps_ms",
        "confirmed_ratio_vs_mps",
        "confirmed_win",
    ]

    var lines = [header.joined(separator: ",")]
    for record in records {
        let fields = [
            record.target.name,
            record.target.shape.description,
            record.bestEvaluation.config.description,
            String(format: "%.6f", record.bestEvaluation.result.medianMilliseconds),
            String(format: "%.6f", record.mpsResult.medianMilliseconds),
            String(format: "%.6f", record.bestEvaluation.result.medianMilliseconds / record.mpsResult.medianMilliseconds),
            String(format: "%.6f", record.bestConfidence.medianOfMedians),
            String(format: "%.6f", record.mpsConfidence.medianOfMedians),
            String(format: "%.6f", record.confirmedRatio),
            record.confirmedWin ? "true" : "false",
        ]
        lines.append(fields.map(csvField).joined(separator: ","))
    }

    try lines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("shape_hunt.csv"),
        atomically: true,
        encoding: .utf8
    )
}

func topValidatedCandidates(
    from entries: [(TiledKernelConfig, BenchmarkResult)],
    count: Int
) -> [(TiledKernelConfig, BenchmarkResult)] {
    let stable = entries.filter { $0.1.isStable(stabilityThreshold: stabilityThreshold) }
    let pool = stable.isEmpty ? entries : stable
    return Array(
        pool.sorted { lhs, rhs in
            let lhsScore = lhs.1.tuningScore(stabilityThreshold: stabilityThreshold)
            let rhsScore = rhs.1.tuningScore(stabilityThreshold: stabilityThreshold)
            if lhsScore == rhsScore {
                return lhs.0.description < rhs.0.description
            }
            return lhsScore < rhsScore
        }
        .prefix(count)
    )
}

do {
    let runner = try MetalComputeRunner()
    let candidateConfigs = runner.tiledConfigs
    print("Using Metal device: \(runner.deviceName)")
    print("Metal capability notes: \(runner.metalCapabilityNotes.joined(separator: ", "))")
    print("SIMD-group matrix path candidate: \(runner.likelySupportsSIMDGroupMatrixPath ? "yes" : "no")")
    print("SIMD-group matrix kernels enabled: \(runner.supportsSIMDGroupMatrixPath ? "yes" : "no")")
    print("Available tiled configs: \(candidateConfigs.map(\.description).joined(separator: ", "))")
    print("")

    // Smoke-test the simplest kernel first so GPU setup bugs show up early.
    print("== Vector Add Correctness ==")
    for length in vectorLengths {
        let ok = try runner.runVectorAdd(length: length)
        print("length \(length): \(ok ? "PASS" : "FAIL")")
    }
    print("")

    print("== Matmul Correctness ==")
    var verifiedConfigs: [TiledKernelConfig] = []
    for shape in correctnessShapes {
        let naiveOK = try runner.verifyMatmul(shape: shape)
        let tiledStatuses = try candidateConfigs.map { config in
            "\(config)=\(try runner.verifyTiledMatmul(shape: shape, config: config) ? "PASS" : "FAIL")"
        }.joined(separator: " | ")
        print("shape \(shape): naive=\(naiveOK ? "PASS" : "FAIL") | \(tiledStatuses)")
    }
    for config in candidateConfigs {
        let allPassed = try correctnessShapes.allSatisfy { shape in
            try runner.verifyTiledMatmul(shape: shape, config: config)
        }
        if allPassed {
            verifiedConfigs.append(config)
        }
    }
    print("Validated configs for tuning: \(verifiedConfigs.map(\.description).joined(separator: ", "))")
    print("")

    print("== Fused Correctness (Matmul + Bias + ReLU) ==")
    var fusedVerifiedConfigs: [TiledKernelConfig] = []
    for shape in correctnessShapes {
        let fusedStatuses = try candidateConfigs.map { config in
            "\(config)=\(try runner.verifyFusedTiledMatmulBiasRelu(shape: shape, config: config) ? "PASS" : "FAIL")"
        }.joined(separator: " | ")
        print("shape \(shape): \(fusedStatuses)")
    }
    for config in candidateConfigs {
        let allPassed = try correctnessShapes.allSatisfy { shape in
            try runner.verifyFusedTiledMatmulBiasRelu(shape: shape, config: config)
        }
        if allPassed {
            fusedVerifiedConfigs.append(config)
        }
    }
    print("Validated fused configs for tuning: \(fusedVerifiedConfigs.map(\.description).joined(separator: ", "))")
    print("")

    // Exhaustively benchmark the small validated config set before introducing SA.
    print("== Deterministic Tuning Sweep ==")
    var sweepRecords: [SweepRecord] = []
    var exhaustiveBestByTarget: [String: ExhaustiveBestRecord] = [:]
    var exhaustiveCacheByTarget: [String: [TiledKernelConfig: BenchmarkResult]] = [:]
    var exhaustiveRanksByTarget: [String: [TiledKernelConfig: Int]] = [:]
    var exhaustiveSortedByTarget: [String: [(TiledKernelConfig, BenchmarkResult)]] = [:]
    var plainTargetSummaries: [PlainTargetSummary] = []
    for (index, target) in tuningTargets.enumerated() {
        let shape = target.shape
        let fixture = runner.makeMatMulBenchmarkFixture(shape: shape, seed: benchmarkSeedBase + UInt64(index))
        let cpu = runner.benchmarkCPUMatmul(fixture: fixture, iterations: benchmarkIterations, warmup: benchmarkWarmup)
        let naiveMetal = try runner.benchmarkMetalMatmul(fixture: fixture, iterations: benchmarkIterations, warmup: benchmarkWarmup)
        let tiledResults = try verifiedConfigs.map { config in
            (config, try runner.benchmarkTiledMetalMatmul(fixture: fixture, config: config, iterations: benchmarkIterations, warmup: benchmarkWarmup))
        }
        let mps = runner.supportsMPS ? try runner.benchmarkMPSMatmul(fixture: fixture, iterations: benchmarkIterations, warmup: benchmarkWarmup) : nil
        print("target \(target.name) [\(target.category)] shape \(shape)")
        printBenchmark(label: "  CPU", shape: shape, result: cpu)
        printBenchmark(label: "  Metal naive", shape: shape, result: naiveMetal)
        sweepRecords.append(SweepRecord(target: target, kernel: "naive", config: nil, result: naiveMetal, tuningScore: nil))
        for (config, result) in tiledResults {
            printBenchmark(label: "  Metal \(config)", shape: shape, result: result)
            sweepRecords.append(SweepRecord(target: target, kernel: "tiled", config: config, result: result, tuningScore: result.tuningScore(stabilityThreshold: stabilityThreshold)))
        }
        if let mps {
            printBenchmark(label: "  MPS", shape: shape, result: mps)
            sweepRecords.append(SweepRecord(target: target, kernel: "mps", config: nil, result: mps, tuningScore: nil))
        } else {
            print("  MPS: unavailable on this device")
        }
        let naiveSpeedup = cpu.medianMilliseconds / naiveMetal.medianMilliseconds
        print("  Metal naive vs CPU speedup: \(String(format: "%.2f", naiveSpeedup))x")
        for (config, result) in tiledResults {
            let speedup = cpu.medianMilliseconds / result.medianMilliseconds
            let versusNaive = naiveMetal.medianMilliseconds / result.medianMilliseconds
            print("  Metal \(config) vs CPU speedup: \(String(format: "%.2f", speedup))x")
            print("  Metal \(config) vs naive speedup: \(String(format: "%.2f", versusNaive))x")
            let stableLabel = result.isStable(stabilityThreshold: stabilityThreshold) ? "stable" : "unstable"
            print("  Metal \(config) tuning score: \(String(format: "%.3f", result.tuningScore(stabilityThreshold: stabilityThreshold))) ms (\(stableLabel))")
        }
        let sortedByScore = tiledResults.sorted { lhs, rhs in
            let lhsScore = lhs.1.tuningScore(stabilityThreshold: stabilityThreshold)
            let rhsScore = rhs.1.tuningScore(stabilityThreshold: stabilityThreshold)
            if lhsScore == rhsScore {
                return lhs.0.description < rhs.0.description
            }
            return lhsScore < rhsScore
        }
        exhaustiveCacheByTarget[target.name] = Dictionary(uniqueKeysWithValues: tiledResults.map { ($0.0, $0.1) })
        exhaustiveSortedByTarget[target.name] = sortedByScore
        exhaustiveRanksByTarget[target.name] = Dictionary(uniqueKeysWithValues: sortedByScore.enumerated().map { (offset, pair) in
            (pair.0, offset + 1)
        })
        if let bestMedian = tiledResults.min(by: { $0.1.medianMilliseconds < $1.1.medianMilliseconds }) {
            print("  Best config by median: \(bestMedian.0) @ \(String(format: "%.3f", bestMedian.1.medianMilliseconds)) ms")
        }
        if let bestScore = sortedByScore.first {
            print("  Best config by tuning score: \(bestScore.0) @ \(String(format: "%.3f", bestScore.1.tuningScore(stabilityThreshold: stabilityThreshold))) ms")
            exhaustiveBestByTarget[target.name] = ExhaustiveBestRecord(
                target: target,
                evaluation: SAEvaluation(
                    config: bestScore.0,
                    result: bestScore.1,
                    score: bestScore.1.tuningScore(stabilityThreshold: stabilityThreshold)
                )
            )

            let bestConfidence = try measureConfidence(passCount: confidencePassCount) {
                try runner.benchmarkTiledMetalMatmul(
                    fixture: fixture,
                    config: bestScore.0,
                    iterations: confidenceIterations,
                    warmup: confidenceWarmup
                )
            }
            printConfidence(label: "  Confidence custom \(bestScore.0)", stats: bestConfidence)

            let mpsConfidence: ConfidenceSummary?
            if runner.supportsMPS {
                mpsConfidence = try measureConfidence(passCount: confidencePassCount) {
                    try runner.benchmarkMPSMatmul(
                        fixture: fixture,
                        iterations: confidenceIterations,
                        warmup: confidenceWarmup
                    )
                }
                if let mpsConfidence {
                    printConfidence(label: "  Confidence MPS", stats: mpsConfidence)
                }
            } else {
                mpsConfidence = nil
            }

            plainTargetSummaries.append(
                PlainTargetSummary(
                    target: target,
                    bestEvaluation: SAEvaluation(
                        config: bestScore.0,
                        result: bestScore.1,
                        score: bestScore.1.tuningScore(stabilityThreshold: stabilityThreshold)
                    ),
                    mpsResult: mps,
                    bestConfidence: bestConfidence,
                    mpsConfidence: mpsConfidence
                )
            )
        }
        if let mps {
            let naiveRatio = naiveMetal.medianMilliseconds / mps.medianMilliseconds
            print("  Metal naive vs MPS latency ratio: \(String(format: "%.2f", naiveRatio))x")
            for (config, result) in tiledResults {
                let ratio = result.medianMilliseconds / mps.medianMilliseconds
                print("  Metal \(config) vs MPS latency ratio: \(String(format: "%.2f", ratio))x")
            }
        }
    }
    try writeSweepResults(sweepRecords)
    try writeConfidenceResults(plainTargetSummaries)
    print("")
    print("== Fused Epilogue Sweep (Matmul + Bias + ReLU) ==")
    var fusedSweepRecords: [FusedSweepRecord] = []
    for (index, target) in fusedTargets.enumerated() {
        let shape = target.shape
        let fixture = runner.makeFusedBenchmarkFixture(shape: shape, seed: fusedBenchmarkSeedBase + UInt64(index))
        let matmulFixture = MatMulBenchmarkFixture(shape: shape, lhs: fixture.lhs, rhs: fixture.rhs)
        let naive = try runner.benchmarkMetalMatmul(fixture: matmulFixture, iterations: benchmarkIterations, warmup: benchmarkWarmup)
        let composed = try fusedVerifiedConfigs.map { config in
            (config, try runner.benchmarkComposedTiledBiasRelu(fixture: fixture, config: config, iterations: benchmarkIterations, warmup: benchmarkWarmup))
        }
        let fused = try fusedVerifiedConfigs.map { config in
            (config, try runner.benchmarkFusedTiledMatmulBiasRelu(fixture: fixture, config: config, iterations: benchmarkIterations, warmup: benchmarkWarmup))
        }
        let mps = runner.supportsMPS ? try runner.benchmarkMPSBiasRelu(fixture: fixture, iterations: benchmarkIterations, warmup: benchmarkWarmup) : nil

        print("target \(target.name) [\(target.category)] shape \(shape)")
        printBenchmark(label: "  Metal naive matmul", shape: shape, result: naive)
        fusedSweepRecords.append(FusedSweepRecord(target: target, kernel: "naive_matmul", config: nil, result: naive, tuningScore: nil))

        for (config, result) in composed {
            printBenchmark(label: "  Composed \(config)", shape: shape, result: result)
            fusedSweepRecords.append(FusedSweepRecord(target: target, kernel: "composed_bias_relu", config: config, result: result, tuningScore: result.tuningScore(stabilityThreshold: stabilityThreshold)))
        }
        for (config, result) in fused {
            printBenchmark(label: "  Fused \(config)", shape: shape, result: result)
            fusedSweepRecords.append(FusedSweepRecord(target: target, kernel: "fused_bias_relu", config: config, result: result, tuningScore: result.tuningScore(stabilityThreshold: stabilityThreshold)))
        }
        if let mps {
            printBenchmark(label: "  MPS + bias+relu", shape: shape, result: mps)
            fusedSweepRecords.append(FusedSweepRecord(target: target, kernel: "mps_bias_relu", config: nil, result: mps, tuningScore: nil))
        } else {
            print("  MPS + bias+relu: unavailable on this device")
        }

        if let bestComposed = composed.min(by: { $0.1.tuningScore(stabilityThreshold: stabilityThreshold) < $1.1.tuningScore(stabilityThreshold: stabilityThreshold) }) {
            print("  Best composed config: \(bestComposed.0) @ \(String(format: "%.3f", bestComposed.1.tuningScore(stabilityThreshold: stabilityThreshold))) ms")
        }
        if let bestFused = fused.min(by: { $0.1.tuningScore(stabilityThreshold: stabilityThreshold) < $1.1.tuningScore(stabilityThreshold: stabilityThreshold) }) {
            print("  Best fused config: \(bestFused.0) @ \(String(format: "%.3f", bestFused.1.tuningScore(stabilityThreshold: stabilityThreshold))) ms")
        }
        if let bestComposed = composed.min(by: { $0.1.tuningScore(stabilityThreshold: stabilityThreshold) < $1.1.tuningScore(stabilityThreshold: stabilityThreshold) }),
           let bestFused = fused.min(by: { $0.1.tuningScore(stabilityThreshold: stabilityThreshold) < $1.1.tuningScore(stabilityThreshold: stabilityThreshold) }) {
            let fusionSpeedup = bestComposed.1.medianMilliseconds / bestFused.1.medianMilliseconds
            print("  Best fused vs best composed speedup: \(String(format: "%.2f", fusionSpeedup))x")
        }
        if let mps,
           let bestFused = fused.min(by: { $0.1.tuningScore(stabilityThreshold: stabilityThreshold) < $1.1.tuningScore(stabilityThreshold: stabilityThreshold) }) {
            let ratio = bestFused.1.medianMilliseconds / mps.medianMilliseconds
            print("  Best fused vs MPS+bias+relu latency ratio: \(String(format: "%.2f", ratio))x")
        }
    }
    try writeFusedSweepResults(fusedSweepRecords)
    print("")
    print("== Simulated Annealing ==")
    var saSummaries: [SASummary] = []
    for (index, target) in tuningTargets.enumerated() {
        guard let exhaustiveBest = exhaustiveBestByTarget[target.name]?.evaluation else {
            continue
        }
        guard let objectiveCache = exhaustiveCacheByTarget[target.name],
              let exhaustiveRanks = exhaustiveRanksByTarget[target.name] else {
            continue
        }

        let summary = try SimulatedAnnealing.run(
            target: target,
            configs: verifiedConfigs,
            exhaustiveBest: exhaustiveBest,
            objectiveCache: objectiveCache,
            exhaustiveRanks: exhaustiveRanks,
            iterations: saIterations,
            stabilityThreshold: stabilityThreshold,
            seed: saSeedBase + UInt64(index)
        )
        saSummaries.append(summary)

        print("target \(target.name) [\(target.category)] shape \(target.shape)")
        print("  SA start config: \(summary.startConfig)")
        print("  SA best config: \(summary.bestEvaluation.config)")
        print("  SA best median: \(String(format: "%.3f", summary.bestEvaluation.result.medianMilliseconds)) ms")
        print("  SA best tuning score: \(String(format: "%.3f", summary.bestEvaluation.score)) ms")
        print("  Exhaustive best: \(summary.exhaustiveBest.config) @ \(String(format: "%.3f", summary.exhaustiveBest.score)) ms")
        print("  SA evaluated configs: \(summary.evaluationCount)")
        print("  SA rank vs exhaustive: \(summary.bestRank)/\(summary.configCount)")
        print("  SA regret vs exhaustive: \(String(format: "%.2f", summary.regretPercent))%")
        print("  SA matched exhaustive best: \(summary.matchedExhaustiveBest ? "yes" : "no")")
        print("  SA score gap vs exhaustive: \(String(format: "%.3f", summary.scoreGapMilliseconds)) ms")
    }
    print("")
    print("== Shape Hunt vs MPS ==")
    var shapeHuntRecords: [ShapeHuntRecord] = []
    var shapeHuntSortedByTarget: [String: [(TiledKernelConfig, BenchmarkResult)]] = [:]
    if runner.supportsMPS {
        for (index, target) in shapeHuntTargets.enumerated() {
            let fixture = runner.makeMatMulBenchmarkFixture(shape: target.shape, seed: benchmarkSeedBase + 0x100 + UInt64(index))
            let tiledResults = try verifiedConfigs.map { config in
                (config, try runner.benchmarkTiledMetalMatmul(fixture: fixture, config: config, iterations: benchmarkIterations, warmup: benchmarkWarmup))
            }
            let mps = try runner.benchmarkMPSMatmul(fixture: fixture, iterations: benchmarkIterations, warmup: benchmarkWarmup)
            let sortedByScore = tiledResults.sorted { lhs, rhs in
                let lhsScore = lhs.1.tuningScore(stabilityThreshold: stabilityThreshold)
                let rhsScore = rhs.1.tuningScore(stabilityThreshold: stabilityThreshold)
                if lhsScore == rhsScore {
                    return lhs.0.description < rhs.0.description
                }
                return lhsScore < rhsScore
            }
            shapeHuntSortedByTarget[target.name] = sortedByScore
            guard let best = sortedByScore.first else {
                continue
            }

            let bestConfidence = try measureConfidence(passCount: confidencePassCount) {
                try runner.benchmarkTiledMetalMatmul(
                    fixture: fixture,
                    config: best.0,
                    iterations: confidenceIterations,
                    warmup: confidenceWarmup
                )
            }
            let mpsConfidence = try measureConfidence(passCount: confidencePassCount) {
                try runner.benchmarkMPSMatmul(
                    fixture: fixture,
                    iterations: confidenceIterations,
                    warmup: confidenceWarmup
                )
            }

            let record = ShapeHuntRecord(
                target: target,
                bestEvaluation: SAEvaluation(
                    config: best.0,
                    result: best.1,
                    score: best.1.tuningScore(stabilityThreshold: stabilityThreshold)
                ),
                mpsResult: mps,
                bestConfidence: bestConfidence,
                mpsConfidence: mpsConfidence
            )
            shapeHuntRecords.append(record)

            print("target \(target.name) shape \(target.shape)")
            print("  Best config by tuning score: \(best.0) @ \(String(format: "%.3f", best.1.tuningScore(stabilityThreshold: stabilityThreshold))) ms")
            print("  Sweep ratio vs MPS: \(String(format: "%.2f", best.1.medianMilliseconds / mps.medianMilliseconds))x")
            printConfidence(label: "  Confidence custom \(best.0)", stats: bestConfidence)
            printConfidence(label: "  Confidence MPS", stats: mpsConfidence)
            print("  Confirmed ratio vs MPS: \(String(format: "%.2f", record.confirmedRatio))x")
            print("  Confirmed win: \(record.confirmedWin ? "yes" : "no")")
        }
    } else {
        print("MPS unavailable on this device, skipping shape hunt.")
    }
    print("")
    print("== Strict Win Validation ==")
    var winValidationSummaries: [WinValidationSummary] = []
    if runner.supportsMPS {
        for targetName in winValidationTargetNames {
            if let target = tuningTargets.first(where: { $0.name == targetName }),
               let sortedResults = exhaustiveSortedByTarget[target.name] {
                let fixture = runner.makeMatMulBenchmarkFixture(
                    shape: target.shape,
                    seed: benchmarkSeedBase + UInt64(tuningTargets.firstIndex(where: { $0.name == target.name }) ?? 0)
                )
                let contenders = topValidatedCandidates(from: sortedResults, count: winValidationTopK)
                let mpsConfidence = try measureConfidence(passCount: winValidationPassCount) {
                    try runner.benchmarkMPSMatmul(
                        fixture: fixture,
                        iterations: winValidationIterations,
                        warmup: winValidationWarmup
                    )
                }
                var candidateSummaries: [WinValidationCandidateSummary] = []
                for (config, result) in contenders {
                    let confidence = try measureConfidence(passCount: winValidationPassCount) {
                        try runner.benchmarkTiledMetalMatmul(
                            fixture: fixture,
                            config: config,
                            iterations: winValidationIterations,
                            warmup: winValidationWarmup
                        )
                    }
                    candidateSummaries.append(
                        WinValidationCandidateSummary(
                            config: config,
                            sweepMedianMilliseconds: result.medianMilliseconds,
                            sweepTuningScoreMilliseconds: result.tuningScore(stabilityThreshold: stabilityThreshold),
                            confidence: confidence
                        )
                    )
                }
                let summary = WinValidationSummary(
                    target: target,
                    mpsConfidence: mpsConfidence,
                    candidates: candidateSummaries.sorted {
                        $0.confidence.medianOfMedians < $1.confidence.medianOfMedians
                    }
                )
                winValidationSummaries.append(summary)
                print("target \(target.name) shape \(target.shape)")
                printConfidence(label: "  Validation MPS", stats: mpsConfidence)
                for candidate in summary.candidates {
                    printConfidence(label: "  Validation \(candidate.config)", stats: candidate.confidence)
                    print("  Validation ratio vs MPS for \(candidate.config): \(String(format: "%.2f", candidate.confirmedRatio(vs: mpsConfidence)))x")
                }
                if let winner = summary.confirmedWinner {
                    print("  Survived stricter rerun: yes, \(winner.config) @ \(String(format: "%.3f", winner.confidence.medianOfMedians)) ms")
                } else {
                    print("  Survived stricter rerun: no")
                }
                continue
            }

            if let target = shapeHuntTargets.first(where: { $0.name == targetName }),
               let sortedResults = shapeHuntSortedByTarget[target.name] {
                let fixture = runner.makeMatMulBenchmarkFixture(
                    shape: target.shape,
                    seed: benchmarkSeedBase + 0x100 + UInt64(shapeHuntTargets.firstIndex(where: { $0.name == target.name }) ?? 0)
                )
                let contenders = topValidatedCandidates(from: sortedResults, count: winValidationTopK)
                let mpsConfidence = try measureConfidence(passCount: winValidationPassCount) {
                    try runner.benchmarkMPSMatmul(
                        fixture: fixture,
                        iterations: winValidationIterations,
                        warmup: winValidationWarmup
                    )
                }
                var candidateSummaries: [WinValidationCandidateSummary] = []
                for (config, result) in contenders {
                    let confidence = try measureConfidence(passCount: winValidationPassCount) {
                        try runner.benchmarkTiledMetalMatmul(
                            fixture: fixture,
                            config: config,
                            iterations: winValidationIterations,
                            warmup: winValidationWarmup
                        )
                    }
                    candidateSummaries.append(
                        WinValidationCandidateSummary(
                            config: config,
                            sweepMedianMilliseconds: result.medianMilliseconds,
                            sweepTuningScoreMilliseconds: result.tuningScore(stabilityThreshold: stabilityThreshold),
                            confidence: confidence
                        )
                    )
                }
                let summary = WinValidationSummary(
                    target: target,
                    mpsConfidence: mpsConfidence,
                    candidates: candidateSummaries.sorted {
                        $0.confidence.medianOfMedians < $1.confidence.medianOfMedians
                    }
                )
                winValidationSummaries.append(summary)
                print("target \(target.name) shape \(target.shape)")
                printConfidence(label: "  Validation MPS", stats: mpsConfidence)
                for candidate in summary.candidates {
                    printConfidence(label: "  Validation \(candidate.config)", stats: candidate.confidence)
                    print("  Validation ratio vs MPS for \(candidate.config): \(String(format: "%.2f", candidate.confirmedRatio(vs: mpsConfidence)))x")
                }
                if let winner = summary.confirmedWinner {
                    print("  Survived stricter rerun: yes, \(winner.config) @ \(String(format: "%.3f", winner.confidence.medianOfMedians)) ms")
                } else {
                    print("  Survived stricter rerun: no")
                }
            }
        }
    } else {
        print("MPS unavailable on this device, skipping strict win validation.")
    }
    try writeSAResults(saSummaries)
    try writeShapeHuntResults(shapeHuntRecords)
    try writeWinValidationResults(winValidationSummaries)
    try writeLatestResultsReport(
        plainSummaries: plainTargetSummaries,
        saSummaries: saSummaries,
        shapeHuntRecords: shapeHuntRecords,
        winValidationSummaries: winValidationSummaries
    )
    try writePlainVsMPSPlot(plainTargetSummaries)
    try writeSARegretPlot(saSummaries)
    print("")
    print("Wrote tuning sweep results to results/tuning_sweep.csv")
    print("Wrote confidence summary to results/confidence_summary.csv")
    print("Wrote fused tuning sweep results to results/fused_tuning_sweep.csv")
    print("Wrote SA summary to results/sa_summary.csv")
    print("Wrote SA history to results/sa_history.csv")
    print("Wrote shape hunt summary to results/shape_hunt.csv")
    print("Wrote strict win validation to results/confirmed_wins.csv")
    print("Wrote latest results report to results/latest_results.md")
    print("Wrote plots to results/plain_vs_mps.svg and results/sa_regret.svg")
} catch {
    fputs("MetalHello failed: \(error)\n", stderr)
    exit(1)
}

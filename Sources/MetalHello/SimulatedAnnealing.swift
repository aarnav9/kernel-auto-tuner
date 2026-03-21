import Foundation

struct SAEvaluation {
    let config: TiledKernelConfig
    let result: BenchmarkResult
    let score: Double
}

struct SAHistoryEntry {
    let iteration: Int
    let temperature: Double
    let currentConfig: TiledKernelConfig
    let candidateConfig: TiledKernelConfig
    let accepted: Bool
    let currentScore: Double
    let bestConfig: TiledKernelConfig
    let bestScore: Double
}

struct SASummary {
    let target: TuningTarget
    let seed: UInt64
    let startConfig: TiledKernelConfig
    let bestEvaluation: SAEvaluation
    let exhaustiveBest: SAEvaluation
    let bestRank: Int
    let configCount: Int
    let evaluationCount: Int
    let history: [SAHistoryEntry]

    var matchedExhaustiveBest: Bool {
        bestEvaluation.config == exhaustiveBest.config
    }

    var scoreGapMilliseconds: Double {
        bestEvaluation.score - exhaustiveBest.score
    }

    var regretPercent: Double {
        guard exhaustiveBest.score > 0 else { return 0 }
        return (bestEvaluation.score / exhaustiveBest.score - 1.0) * 100.0
    }
}

struct SeededGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func nextDouble() -> Double {
        Double(nextUInt64() >> 11) / Double(1 << 53)
    }

    mutating func pickIndex(count: Int) -> Int {
        Int(nextUInt64() % UInt64(count))
    }
}

enum SimulatedAnnealing {
    static func run(
        target: TuningTarget,
        configs: [TiledKernelConfig],
        exhaustiveBest: SAEvaluation,
        objectiveCache: [TiledKernelConfig: BenchmarkResult],
        exhaustiveRanks: [TiledKernelConfig: Int],
        iterations: Int,
        stabilityThreshold: Double,
        seed: UInt64
    ) throws -> SASummary {
        precondition(!configs.isEmpty, "SA needs at least one candidate config.")

        var rng = SeededGenerator(seed: seed)
        let startConfig = configs[rng.pickIndex(count: configs.count)]
        var visitedConfigs: Set<TiledKernelConfig> = []

        func evaluate(_ config: TiledKernelConfig) throws -> SAEvaluation {
            guard let result = objectiveCache[config] else {
                throw NSError(domain: "SimulatedAnnealing", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "Missing cached objective for config \(config) on target \(target.name)."
                ])
            }
            visitedConfigs.insert(config)
            return SAEvaluation(
                config: config,
                result: result,
                score: result.tuningScore(stabilityThreshold: stabilityThreshold)
            )
        }

        var current = try evaluate(startConfig)
        var best = current
        var history: [SAHistoryEntry] = []
        history.reserveCapacity(iterations)

        for iteration in 0..<iterations {
            let temperature = scheduleTemperature(iteration: iteration, totalIterations: iterations, baselineScore: current.score)
            let candidatePool = nearestNeighbors(for: current.config, within: configs, count: min(4, max(1, configs.count - 1)))
            let candidateConfig = candidatePool.isEmpty ? current.config : candidatePool[rng.pickIndex(count: candidatePool.count)]
            let candidate = try evaluate(candidateConfig)

            let delta = candidate.score - current.score
            let accept = delta <= 0 || rng.nextDouble() < exp(-delta / max(temperature, 0.0001))
            if accept {
                current = candidate
            }
            if current.score < best.score {
                best = current
            }

            history.append(
                SAHistoryEntry(
                    iteration: iteration,
                    temperature: temperature,
                    currentConfig: current.config,
                    candidateConfig: candidateConfig,
                    accepted: accept,
                    currentScore: current.score,
                    bestConfig: best.config,
                    bestScore: best.score
                )
            )
        }

        return SASummary(
            target: target,
            seed: seed,
            startConfig: startConfig,
            bestEvaluation: best,
            exhaustiveBest: exhaustiveBest,
            bestRank: exhaustiveRanks[best.config] ?? (configs.count + 1),
            configCount: configs.count,
            evaluationCount: visitedConfigs.count,
            history: history
        )
    }

    private static func scheduleTemperature(iteration: Int, totalIterations: Int, baselineScore: Double) -> Double {
        let start = max(0.02, baselineScore * 0.35)
        let end = max(0.002, start * 0.08)
        let progress = Double(iteration + 1) / Double(max(totalIterations, 1))
        return start * pow(end / start, progress)
    }

    private static func nearestNeighbors(
        for config: TiledKernelConfig,
        within allConfigs: [TiledKernelConfig],
        count: Int
    ) -> [TiledKernelConfig] {
        allConfigs
            .filter { $0 != config }
            .sorted { configDistance(config, $0) < configDistance(config, $1) }
            .prefix(count)
            .map { $0 }
    }

    private static func configDistance(_ lhs: TiledKernelConfig, _ rhs: TiledKernelConfig) -> Double {
        func logDistance(_ left: Int, _ right: Int) -> Double {
            abs(log2(Double(left)) - log2(Double(right)))
        }

        return
            logDistance(lhs.tileM, rhs.tileM) +
            logDistance(lhs.tileN, rhs.tileN) +
            logDistance(lhs.tileK, rhs.tileK) +
            logDistance(lhs.threadsX, rhs.threadsX) +
            logDistance(lhs.threadsY, rhs.threadsY)
    }
}

import Foundation

struct ConfidenceSummary {
    let passMedians: [Double]
    let passScores: [Double]

    var passCount: Int { passMedians.count }
    var medianOfMedians: Double { Self.median(passMedians) }
    var medianOfScores: Double { Self.median(passScores) }
    var minMedian: Double { passMedians.min() ?? 0 }
    var maxMedian: Double { passMedians.max() ?? 0 }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        if sorted.count.isMultiple(of: 2) {
            let upper = sorted.count / 2
            return (sorted[upper - 1] + sorted[upper]) / 2.0
        }
        return sorted[sorted.count / 2]
    }
}

struct PlainTargetSummary {
    let target: TuningTarget
    let bestEvaluation: SAEvaluation
    let mpsResult: BenchmarkResult?
    let bestConfidence: ConfidenceSummary
    let mpsConfidence: ConfidenceSummary?

    var mpsLatencyRatio: Double? {
        guard let mpsResult, mpsResult.medianMilliseconds > 0 else { return nil }
        return bestEvaluation.result.medianMilliseconds / mpsResult.medianMilliseconds
    }

    var verdict: String {
        guard let ratio = mpsLatencyRatio else { return "no_mps" }
        if ratio <= 1.0 {
            return "beat"
        }
        if ratio <= 1.10 {
            return "near_match"
        }
        return "trail"
    }
}

struct WinValidationCandidateSummary {
    let config: TiledKernelConfig
    let sweepMedianMilliseconds: Double
    let sweepTuningScoreMilliseconds: Double
    let confidence: ConfidenceSummary

    func confirmedRatio(vs mps: ConfidenceSummary) -> Double {
        confidence.medianOfMedians / mps.medianOfMedians
    }

    func confirmedWin(vs mps: ConfidenceSummary) -> Bool {
        confirmedRatio(vs: mps) < 1.0
    }
}

struct WinValidationSummary {
    let target: TuningTarget
    let mpsConfidence: ConfidenceSummary
    let candidates: [WinValidationCandidateSummary]

    var validatedBest: WinValidationCandidateSummary? {
        candidates.min { lhs, rhs in
            lhs.confidence.medianOfMedians < rhs.confidence.medianOfMedians
        }
    }

    var confirmedWinner: WinValidationCandidateSummary? {
        candidates
            .filter { $0.confirmedWin(vs: mpsConfidence) }
            .min { lhs, rhs in
                lhs.confidence.medianOfMedians < rhs.confidence.medianOfMedians
            }
    }
}

func measureConfidence(passCount: Int, benchmark: () throws -> BenchmarkResult) throws -> ConfidenceSummary {
    var medians: [Double] = []
    var scores: [Double] = []
    medians.reserveCapacity(passCount)
    scores.reserveCapacity(passCount)

    for _ in 0..<passCount {
        let result = try benchmark()
        medians.append(result.medianMilliseconds)
        scores.append(result.tuningScore())
    }

    return ConfidenceSummary(passMedians: medians, passScores: scores)
}

func printConfidence(label: String, stats: ConfidenceSummary) {
    let medians = stats.passMedians.map { String(format: "%.3f", $0) }.joined(separator: ", ")
    print("\(label): median_of_medians=\(String(format: "%.3f", stats.medianOfMedians)) ms | min=\(String(format: "%.3f", stats.minMedian)) ms | max=\(String(format: "%.3f", stats.maxMedian)) ms | passes=[\(medians)]")
}

func writeConfidenceResults(_ summaries: [PlainTargetSummary]) throws {
    let resultsDir = try ensureResultsDirectory()
    let header = [
        "target",
        "category",
        "shape",
        "kernel",
        "config",
        "pass_count",
        "median_of_medians_ms",
        "min_median_ms",
        "max_median_ms",
        "pass_medians_ms",
    ]

    var lines = [header.joined(separator: ",")]
    for summary in summaries {
        let bestFields = [
            summary.target.name,
            summary.target.category,
            summary.target.shape.description,
            "best_custom",
            summary.bestEvaluation.config.description,
            String(summary.bestConfidence.passCount),
            String(format: "%.6f", summary.bestConfidence.medianOfMedians),
            String(format: "%.6f", summary.bestConfidence.minMedian),
            String(format: "%.6f", summary.bestConfidence.maxMedian),
            summary.bestConfidence.passMedians.map { String(format: "%.6f", $0) }.joined(separator: ";"),
        ]
        lines.append(bestFields.map(csvField).joined(separator: ","))

        if let mpsConfidence = summary.mpsConfidence {
            let mpsFields = [
                summary.target.name,
                summary.target.category,
                summary.target.shape.description,
                "mps",
                "",
                String(mpsConfidence.passCount),
                String(format: "%.6f", mpsConfidence.medianOfMedians),
                String(format: "%.6f", mpsConfidence.minMedian),
                String(format: "%.6f", mpsConfidence.maxMedian),
                mpsConfidence.passMedians.map { String(format: "%.6f", $0) }.joined(separator: ";"),
            ]
            lines.append(mpsFields.map(csvField).joined(separator: ","))
        }
    }

    try lines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("confidence_summary.csv"),
        atomically: true,
        encoding: .utf8
    )
}

func writeWinValidationResults(_ summaries: [WinValidationSummary]) throws {
    let resultsDir = try ensureResultsDirectory()
    let header = [
        "target",
        "category",
        "shape",
        "config",
        "sweep_median_ms",
        "sweep_tuning_score_ms",
        "validation_pass_count",
        "validation_median_of_medians_ms",
        "validation_min_median_ms",
        "validation_max_median_ms",
        "mps_validation_median_of_medians_ms",
        "confirmed_ratio_vs_mps",
        "confirmed_win",
        "selected_for_claim",
    ]

    var lines = [header.joined(separator: ",")]
    for summary in summaries {
        let selectedConfig = summary.confirmedWinner?.config
        for candidate in summary.candidates {
            let fields = [
                summary.target.name,
                summary.target.category,
                summary.target.shape.description,
                candidate.config.description,
                String(format: "%.6f", candidate.sweepMedianMilliseconds),
                String(format: "%.6f", candidate.sweepTuningScoreMilliseconds),
                String(candidate.confidence.passCount),
                String(format: "%.6f", candidate.confidence.medianOfMedians),
                String(format: "%.6f", candidate.confidence.minMedian),
                String(format: "%.6f", candidate.confidence.maxMedian),
                String(format: "%.6f", summary.mpsConfidence.medianOfMedians),
                String(format: "%.6f", candidate.confirmedRatio(vs: summary.mpsConfidence)),
                candidate.confirmedWin(vs: summary.mpsConfidence) ? "true" : "false",
                candidate.config == selectedConfig ? "true" : "false",
            ]
            lines.append(fields.map(csvField).joined(separator: ","))
        }
    }

    try lines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("confirmed_wins.csv"),
        atomically: true,
        encoding: .utf8
    )
}

func writeLatestResultsReport(
    plainSummaries: [PlainTargetSummary],
    saSummaries: [SASummary],
    shapeHuntRecords: [ShapeHuntRecord],
    winValidationSummaries: [WinValidationSummary]
) throws {
    let resultsDir = try ensureResultsDirectory()

    let beatTargets = plainSummaries.filter { $0.verdict == "beat" }.map(\.target.name)
    let nearMatchTargets = plainSummaries.filter { $0.verdict == "near_match" }.map(\.target.name)

    var lines: [String] = []
    lines.append("# Latest Results")
    lines.append("")
    lines.append("This report is generated by `swift run MetalHello` from the latest local benchmark run.")
    lines.append("")
    lines.append("## Plain GEMM Highlights")
    lines.append("")
    lines.append("- Targets beating `MPS` by median latency: \(beatTargets.isEmpty ? "none in this run" : beatTargets.joined(separator: ", "))")
    lines.append("- Targets within 10% of `MPS`: \(nearMatchTargets.isEmpty ? "none in this run" : nearMatchTargets.joined(separator: ", "))")
    lines.append("- Targets still trailing `MPS`: \(plainSummaries.filter { $0.verdict == "trail" }.map(\.target.name).joined(separator: ", "))")
    lines.append("")
    lines.append("| target | best config | best median ms | MPS median ms | ratio vs MPS | verdict | confidence median-of-medians | confidence range |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | --- |")
    for summary in plainSummaries {
        let mpsMedian = summary.mpsResult.map { String(format: "%.3f", $0.medianMilliseconds) } ?? "n/a"
        let ratio = summary.mpsLatencyRatio.map { String(format: "%.2f", $0) } ?? "n/a"
        let range = "\(String(format: "%.3f", summary.bestConfidence.minMedian))-\(String(format: "%.3f", summary.bestConfidence.maxMedian))"
        lines.append("| \(summary.target.name) | `\(summary.bestEvaluation.config)` | \(String(format: "%.3f", summary.bestEvaluation.result.medianMilliseconds)) | \(mpsMedian) | \(ratio)x | \(summary.verdict) | \(String(format: "%.3f", summary.bestConfidence.medianOfMedians)) | \(range) |")
    }
    lines.append("")
    lines.append("## Simulated Annealing")
    lines.append("")
    lines.append("| target | SA best config | exhaustive best config | rank | regret | configs evaluated |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for summary in saSummaries {
        lines.append("| \(summary.target.name) | `\(summary.bestEvaluation.config)` | `\(summary.exhaustiveBest.config)` | \(summary.bestRank)/\(summary.configCount) | \(String(format: "%.2f", summary.regretPercent))% | \(summary.evaluationCount) |")
    }
    lines.append("")
    lines.append("## Shape Hunt")
    lines.append("")
    let confirmedWins = shapeHuntRecords.filter(\.confirmedWin)
    lines.append("- Confirmed custom-over-`MPS` wins: \(confirmedWins.isEmpty ? "none in this run" : confirmedWins.map(\.target.name).joined(separator: ", "))")
    lines.append("")
    lines.append("| target | best config | confidence custom ms | confidence MPS ms | confirmed ratio | win |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for record in shapeHuntRecords {
        lines.append("| \(record.target.name) | `\(record.bestEvaluation.config)` | \(String(format: "%.3f", record.bestConfidence.medianOfMedians)) | \(String(format: "%.3f", record.mpsConfidence.medianOfMedians)) | \(String(format: "%.2f", record.confirmedRatio))x | \(record.confirmedWin ? "yes" : "no") |")
    }
    lines.append("")
    lines.append("## Strict Win Validation")
    lines.append("")
    let publicClaimWins = winValidationSummaries.compactMap { summary -> String? in
        guard let winner = summary.confirmedWinner else { return nil }
        return "\(summary.target.name) (`\(winner.config)`)"
    }
    lines.append("- Wins that survived stricter reruns: \(publicClaimWins.isEmpty ? "none in this run" : publicClaimWins.joined(separator: ", "))")
    lines.append("")
    lines.append("| target | selected config | custom validation ms | MPS validation ms | confirmed ratio | public-claim safe |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for summary in winValidationSummaries {
        if let winner = summary.confirmedWinner {
            lines.append("| \(summary.target.name) | `\(winner.config)` | \(String(format: "%.3f", winner.confidence.medianOfMedians)) | \(String(format: "%.3f", summary.mpsConfidence.medianOfMedians)) | \(String(format: "%.2f", winner.confirmedRatio(vs: summary.mpsConfidence)))x | yes |")
        } else if let best = summary.validatedBest {
            lines.append("| \(summary.target.name) | `\(best.config)` | \(String(format: "%.3f", best.confidence.medianOfMedians)) | \(String(format: "%.3f", summary.mpsConfidence.medianOfMedians)) | \(String(format: "%.2f", best.confirmedRatio(vs: summary.mpsConfidence)))x | no |")
        }
    }
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("- Best custom config vs `MPS`: [results/plain_vs_mps.svg](/Users/aarnav/Downloads/kernel-auto-tuner/results/plain_vs_mps.svg)")
    lines.append("- SA regret and rank: [results/sa_regret.svg](/Users/aarnav/Downloads/kernel-auto-tuner/results/sa_regret.svg)")

    try lines.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("latest_results.md"),
        atomically: true,
        encoding: .utf8
    )
}

func writePlainVsMPSPlot(_ summaries: [PlainTargetSummary]) throws {
    let resultsDir = try ensureResultsDirectory()
    let points = summaries.filter { $0.mpsResult != nil }
    let width = 760
    let rowHeight = 58
    let height = 90 + rowHeight * max(points.count, 1)
    let leftMargin = 170.0
    let rightMargin = 40.0
    let plotWidth = Double(width) - leftMargin - rightMargin
    let maxRatio = max(1.9, points.compactMap(\.mpsLatencyRatio).max() ?? 1.1)
    _ = height

    var svg: [String] = []
    svg.append(##"<svg xmlns="http://www.w3.org/2000/svg" width="\#(width)" height="\#(height)" viewBox="0 0 \#(width) \#(height)">"##)
    svg.append(##"<style>text{font-family:Menlo,monospace;font-size:12px;fill:#1b1b1b}.title{font-size:16px;font-weight:700}.axis{stroke:#666;stroke-width:1}.mps{fill:#d2d8de}.custom{fill:#1f7a8c}.line{stroke:#999;stroke-dasharray:4 4}.note{font-size:11px;fill:#555}</style>"##)
    svg.append(##"<rect width="100%" height="100%" fill="#fffdf8"/>"##)
    svg.append(##"<text class="title" x="24" y="28">Best plain custom config vs MPS (median latency ratio)</text>"##)
    svg.append(##"<text class="note" x="24" y="48">Bars left of 1.0 indicate a custom config beating MPS. Lower is better.</text>"##)

    let baselineX = leftMargin + plotWidth / maxRatio
    _ = baselineX
    svg.append(##"<line class="line" x1="\#(baselineX)" y1="62" x2="\#(baselineX)" y2="\#(height - 18)"/>"##)
    svg.append(##"<text x="\#(baselineX - 10)" y="74">1.0</text>"##)

    for (index, summary) in points.enumerated() {
        let y = 95.0 + Double(index * rowHeight)
        let ratio = summary.mpsLatencyRatio ?? 0
        let barWidth = plotWidth * min(ratio, maxRatio) / maxRatio
        let barColor = ratio <= 1.0 ? "#2a9d8f" : "#c96b3b"
        _ = y
        _ = barWidth
        _ = barColor
        svg.append(##"<text x="24" y="\#(y + 14)">\#(summary.target.name)</text>"##)
        svg.append(##"<rect x="\#(leftMargin)" y="\#(y)" width="\#(plotWidth / maxRatio)" height="18" fill="#e6ebf0"/>"##)
        svg.append(##"<rect x="\#(leftMargin)" y="\#(y)" width="\#(barWidth)" height="18" fill="\#(barColor)"/>"##)
        svg.append(##"<text x="\#(leftMargin + barWidth + 8)" y="\#(y + 14)">\#(String(format: "%.2fx", ratio))</text>"##)
        svg.append(##"<text class="note" x="\#(leftMargin)" y="\#(y + 34)">best=\#(String(format: "%.3f", summary.bestEvaluation.result.medianMilliseconds)) ms | mps=\#(String(format: "%.3f", summary.mpsResult!.medianMilliseconds)) ms</text>"##)
    }

    svg.append("</svg>")
    try svg.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("plain_vs_mps.svg"),
        atomically: true,
        encoding: .utf8
    )
}

func writeSARegretPlot(_ summaries: [SASummary]) throws {
    let resultsDir = try ensureResultsDirectory()
    let width = 760
    let rowHeight = 56
    let height = 90 + rowHeight * max(summaries.count, 1)
    let leftMargin = 170.0
    let rightMargin = 40.0
    let plotWidth = Double(width) - leftMargin - rightMargin
    let maxRegret = max(1.0, summaries.map { max($0.regretPercent, 0) }.max() ?? 1.0)
    _ = height

    var svg: [String] = []
    svg.append(##"<svg xmlns="http://www.w3.org/2000/svg" width="\#(width)" height="\#(height)" viewBox="0 0 \#(width) \#(height)">"##)
    svg.append(##"<style>text{font-family:Menlo,monospace;font-size:12px;fill:#1b1b1b}.title{font-size:16px;font-weight:700}.bar{fill:#7c6ccf}.note{font-size:11px;fill:#555}</style>"##)
    svg.append(##"<rect width="100%" height="100%" fill="#fffdf8"/>"##)
    svg.append(##"<text class="title" x="24" y="28">SA regret vs exhaustive baseline</text>"##)
    svg.append(##"<text class="note" x="24" y="48">Regret is relative to the exhaustive best tuning score. Lower is better.</text>"##)

    for (index, summary) in summaries.enumerated() {
        let y = 95.0 + Double(index * rowHeight)
        let regret = max(summary.regretPercent, 0)
        let barWidth = plotWidth * regret / maxRegret
        _ = y
        _ = barWidth
        svg.append(##"<text x="24" y="\#(y + 14)">\#(summary.target.name)</text>"##)
        svg.append(##"<rect x="\#(leftMargin)" y="\#(y)" width="\#(barWidth)" height="18" class="bar"/>"##)
        svg.append(##"<text x="\#(leftMargin + barWidth + 8)" y="\#(y + 14)">\#(String(format: "%.2f%%", summary.regretPercent))</text>"##)
        svg.append(##"<text class="note" x="\#(leftMargin)" y="\#(y + 34)">rank \#(summary.bestRank)/\#(summary.configCount) | evaluated \#(summary.evaluationCount) configs</text>"##)
    }

    svg.append("</svg>")
    try svg.joined(separator: "\n").write(
        to: resultsDir.appendingPathComponent("sa_regret.svg"),
        atomically: true,
        encoding: .utf8
    )
}

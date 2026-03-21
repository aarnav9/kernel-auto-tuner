import Foundation
import Metal
import MetalPerformanceShaders

enum MetalHelloError: Error, CustomStringConvertible {
    case deviceUnavailable
    case resourceMissing(String)
    case functionMissing(String)
    case pipelineCreationFailed(String)
    case bufferCreationFailed(String)
    case commandEncodingFailed(String)

    var description: String {
        switch self {
        case .deviceUnavailable:
            return "Metal device unavailable on this machine."
        case .resourceMissing(let name):
            return "Unable to find bundled resource \(name)."
        case .functionMissing(let name):
            return "Unable to find Metal function \(name)."
        case .pipelineCreationFailed(let message):
            return "Unable to create compute pipeline: \(message)"
        case .bufferCreationFailed(let name):
            return "Unable to create Metal buffer for \(name)."
        case .commandEncodingFailed(let message):
            return "Unable to encode Metal commands: \(message)"
        }
    }
}

struct MatMulShape: CustomStringConvertible {
    let m: Int
    let n: Int
    let k: Int

    var description: String {
        "\(m)x\(n)x\(k)"
    }
}

struct BenchmarkResult {
    let averageMilliseconds: Double
    let medianMilliseconds: Double
    let minMilliseconds: Double
    let maxMilliseconds: Double
    let samples: [Double]

    var stabilityRatio: Double {
        guard medianMilliseconds > 0 else { return .infinity }
        return maxMilliseconds / medianMilliseconds
    }

    func tuningScore(stabilityThreshold: Double = 1.5) -> Double {
        if stabilityRatio <= stabilityThreshold {
            return medianMilliseconds
        }
        return medianMilliseconds * stabilityRatio
    }

    func isStable(stabilityThreshold: Double = 1.5) -> Bool {
        stabilityRatio <= stabilityThreshold
    }
}

struct MatMulBenchmarkFixture {
    let shape: MatMulShape
    let lhs: [Float]
    let rhs: [Float]
}

struct FusedBenchmarkFixture {
    let shape: MatMulShape
    let lhs: [Float]
    let rhs: [Float]
    let bias: [Float]
}

enum KernelFamily: String, Hashable {
    case scalarTiled = "scalar"
    case simdgroupMatrix = "simdgroup"
}

struct TiledKernelConfig: CustomStringConvertible, Hashable {
    let family: KernelFamily
    let tileM: Int
    let tileN: Int
    let tileK: Int
    let threadsX: Int
    let threadsY: Int

    init(
        family: KernelFamily = .scalarTiled,
        tileM: Int,
        tileN: Int,
        tileK: Int,
        threadsX: Int,
        threadsY: Int
    ) {
        self.family = family
        self.tileM = tileM
        self.tileN = tileN
        self.tileK = tileK
        self.threadsX = threadsX
        self.threadsY = threadsY
    }

    var kernelName: String {
        switch family {
        case .scalarTiled:
            return "matmul_tiled_m\(tileM)_n\(tileN)_k\(tileK)_tx\(threadsX)_ty\(threadsY)"
        case .simdgroupMatrix:
            return "matmul_simdgroup_m\(tileM)_n\(tileN)_k\(tileK)_tx\(threadsX)_ty\(threadsY)"
        }
    }

    var fusedKernelName: String {
        switch family {
        case .scalarTiled:
            return "matmul_bias_relu_tiled_m\(tileM)_n\(tileN)_k\(tileK)_tx\(threadsX)_ty\(threadsY)"
        case .simdgroupMatrix:
            return "matmul_bias_relu_simdgroup_m\(tileM)_n\(tileN)_k\(tileK)_tx\(threadsX)_ty\(threadsY)"
        }
    }

    var outputTileM: Int {
        switch family {
        case .scalarTiled:
            return threadsY
        case .simdgroupMatrix:
            return tileM
        }
    }

    var outputTileN: Int {
        switch family {
        case .scalarTiled:
            return threadsX
        case .simdgroupMatrix:
            return tileN
        }
    }

    var description: String {
        switch family {
        case .scalarTiled:
            return "m\(tileM)n\(tileN)k\(tileK)_t\(threadsX)x\(threadsY)"
        case .simdgroupMatrix:
            return "sg_m\(tileM)n\(tileN)k\(tileK)_t\(threadsX)x\(threadsY)"
        }
    }
}

private struct MatMulParams {
    var M: UInt32
    var N: UInt32
    var K: UInt32
}

private struct EpilogueParams {
    var M: UInt32
    var N: UInt32
}

private struct FixtureSeededGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0xD1B54A32D192ED03 : seed
    }

    mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let unit = Float(Double(nextUInt64() >> 11) / Double(1 << 53))
        return range.lowerBound + (range.upperBound - range.lowerBound) * unit
    }
}

private struct LibraryBuild {
    let library: MTLLibrary
    let includedSIMDGroupSource: Bool
}

final class MetalComputeRunner {
    private static let requestedTiledConfigs = [
        TiledKernelConfig(tileM: 8, tileN: 8, tileK: 8, threadsX: 8, threadsY: 8),
        TiledKernelConfig(tileM: 8, tileN: 16, tileK: 8, threadsX: 16, threadsY: 8),
        TiledKernelConfig(tileM: 16, tileN: 8, tileK: 8, threadsX: 8, threadsY: 16),
        TiledKernelConfig(tileM: 16, tileN: 16, tileK: 8, threadsX: 16, threadsY: 16),
        TiledKernelConfig(tileM: 16, tileN: 16, tileK: 16, threadsX: 16, threadsY: 16),
        TiledKernelConfig(tileM: 8, tileN: 32, tileK: 8, threadsX: 32, threadsY: 8),
        TiledKernelConfig(tileM: 32, tileN: 8, tileK: 8, threadsX: 8, threadsY: 32),
        TiledKernelConfig(tileM: 16, tileN: 32, tileK: 8, threadsX: 32, threadsY: 16),
        TiledKernelConfig(tileM: 32, tileN: 16, tileK: 8, threadsX: 16, threadsY: 32),
        TiledKernelConfig(tileM: 32, tileN: 32, tileK: 8, threadsX: 32, threadsY: 32),
        TiledKernelConfig(tileM: 8, tileN: 64, tileK: 8, threadsX: 64, threadsY: 8),
        TiledKernelConfig(tileM: 64, tileN: 8, tileK: 8, threadsX: 8, threadsY: 64),
        TiledKernelConfig(tileM: 16, tileN: 64, tileK: 8, threadsX: 64, threadsY: 16),
        TiledKernelConfig(tileM: 64, tileN: 16, tileK: 8, threadsX: 16, threadsY: 64),
        TiledKernelConfig(tileM: 16, tileN: 16, tileK: 4, threadsX: 16, threadsY: 16),
        TiledKernelConfig(tileM: 32, tileN: 8, tileK: 4, threadsX: 8, threadsY: 32),
        TiledKernelConfig(family: .simdgroupMatrix, tileM: 16, tileN: 16, tileK: 8, threadsX: 32, threadsY: 4),
        TiledKernelConfig(family: .simdgroupMatrix, tileM: 16, tileN: 16, tileK: 8, threadsX: 16, threadsY: 8),
        TiledKernelConfig(family: .simdgroupMatrix, tileM: 16, tileN: 16, tileK: 8, threadsX: 8, threadsY: 16),
    ]

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let addPipeline: MTLComputePipelineState
    private let addBiasReluPipeline: MTLComputePipelineState
    private let matmulPipeline: MTLComputePipelineState
    private let tiledMatmulPipelines: [TiledKernelConfig: MTLComputePipelineState]
    private let fusedTiledMatmulPipelines: [TiledKernelConfig: MTLComputePipelineState]
    private let simdgroupMatrixSourceEnabled: Bool

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalHelloError.deviceUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalHelloError.commandEncodingFailed("Unable to create command queue.")
        }

        let libraryBuild = try Self.makeLibrary(
            device: device,
            includeSIMDGroupSource: Self.prefersSIMDGroupMatrixSource(on: device)
        )
        let library = libraryBuild.library
        self.device = device
        self.commandQueue = commandQueue
        self.addPipeline = try Self.makePipeline(named: "add_arrays", in: library, device: device)
        self.addBiasReluPipeline = try Self.makePipeline(named: "add_bias_relu", in: library, device: device)
        self.matmulPipeline = try Self.makePipeline(named: "matmul_naive", in: library, device: device)
        self.simdgroupMatrixSourceEnabled = libraryBuild.includedSIMDGroupSource

        var tiledPipelines: [TiledKernelConfig: MTLComputePipelineState] = [:]
        var fusedPipelines: [TiledKernelConfig: MTLComputePipelineState] = [:]
        for config in Self.requestedTiledConfigs {
            guard let tiledPipeline = try Self.makeOptionalPipeline(named: config.kernelName, in: library, device: device),
                  let fusedPipeline = try Self.makeOptionalPipeline(named: config.fusedKernelName, in: library, device: device) else {
                continue
            }
            if config.threadsX * config.threadsY <= tiledPipeline.maxTotalThreadsPerThreadgroup {
                tiledPipelines[config] = tiledPipeline
            }
            if config.threadsX * config.threadsY <= fusedPipeline.maxTotalThreadsPerThreadgroup {
                fusedPipelines[config] = fusedPipeline
            }
        }
        self.tiledMatmulPipelines = tiledPipelines
        self.fusedTiledMatmulPipelines = fusedPipelines
    }

    var deviceName: String {
        device.name
    }

    var supportsMPS: Bool {
        MPSSupportsMTLDevice(device)
    }

    var metalCapabilityNotes: [String] {
        var notes: [String] = []
        if #available(macOS 12.0, *) {
            if device.supportsFamily(.apple8) {
                notes.append("apple8")
            } else if device.supportsFamily(.apple7) {
                notes.append("apple7")
            }
            if device.supportsFamily(.mac2) {
                notes.append("mac2")
            }
        }
        if notes.isEmpty {
            notes.append("unknown_family")
        }
        notes.append("simdgroup_matrix_source=\(simdgroupMatrixSourceEnabled ? "enabled" : "disabled")")
        return notes
    }

    var likelySupportsSIMDGroupMatrixPath: Bool {
        if #available(macOS 12.0, *) {
            return device.supportsFamily(.apple7) || device.supportsFamily(.apple8)
        }
        return false
    }

    var supportsSIMDGroupMatrixPath: Bool {
        likelySupportsSIMDGroupMatrixPath && simdgroupMatrixSourceEnabled
    }

    var tiledConfigs: [TiledKernelConfig] {
        Self.requestedTiledConfigs.filter {
            tiledMatmulPipelines[$0] != nil && fusedTiledMatmulPipelines[$0] != nil
        }
    }

    func makeMatMulBenchmarkFixture(shape: MatMulShape, seed: UInt64) -> MatMulBenchmarkFixture {
        var generator = FixtureSeededGenerator(seed: seed)
        return MatMulBenchmarkFixture(
            shape: shape,
            lhs: Self.seededArray(count: shape.m * shape.k, using: &generator),
            rhs: Self.seededArray(count: shape.k * shape.n, using: &generator)
        )
    }

    func makeFusedBenchmarkFixture(shape: MatMulShape, seed: UInt64) -> FusedBenchmarkFixture {
        var generator = FixtureSeededGenerator(seed: seed)
        return FusedBenchmarkFixture(
            shape: shape,
            lhs: Self.seededArray(count: shape.m * shape.k, using: &generator),
            rhs: Self.seededArray(count: shape.k * shape.n, using: &generator),
            bias: Self.seededArray(count: shape.n, using: &generator)
        )
    }

    func runVectorAdd(length: Int) throws -> Bool {
        let lhs = Self.randomArray(count: length)
        let rhs = Self.randomArray(count: length)
        let gpu = try vectorAdd(lhs: lhs, rhs: rhs)
        let expected = zip(lhs, rhs).map(+)
        return Self.allClose(gpu, expected, tolerance: 1e-6)
    }

    func vectorAdd(lhs: [Float], rhs: [Float]) throws -> [Float] {
        precondition(lhs.count == rhs.count, "Vector sizes must match.")

        let count = lhs.count
        let byteCount = count * MemoryLayout<Float>.stride

        let lhsBuffer = try makeBuffer(from: lhs, label: "vector lhs")
        let rhsBuffer = try makeBuffer(from: rhs, label: "vector rhs")
        let resultBuffer = try makeEmptyBuffer(length: byteCount, label: "vector result")
        let lengthValue = UInt32(count)
        let lengthBuffer = try makeBuffer(from: [lengthValue], label: "vector length")

        try runCommand(label: "vector add") { encoder in
            encoder.setComputePipelineState(addPipeline)
            // Buffer indices must match the shader signature in ComputeKernels.metal.
            encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
            encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer, offset: 0, index: 2)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 3)

            let width = min(addPipeline.maxTotalThreadsPerThreadgroup, max(addPipeline.threadExecutionWidth, 1))
            let threadsPerGroup = MTLSize(width: width, height: 1, depth: 1)
            let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        }

        let pointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func matmul(shape: MatMulShape, lhs: [Float], rhs: [Float]) throws -> [Float] {
        precondition(lhs.count == shape.m * shape.k, "lhs size must match MxK.")
        precondition(rhs.count == shape.k * shape.n, "rhs size must match KxN.")

        let lhsBuffer = try makeBuffer(from: lhs, label: "matmul lhs")
        let rhsBuffer = try makeBuffer(from: rhs, label: "matmul rhs")
        let resultBuffer = try makeEmptyBuffer(
            length: shape.m * shape.n * MemoryLayout<Float>.stride,
            label: "matmul result"
        )
        let params = MatMulParams(M: UInt32(shape.m), N: UInt32(shape.n), K: UInt32(shape.k))
        let paramsBuffer = try makeBuffer(from: [params], label: "matmul params")

        try runCommand(label: "matmul \(shape)") { encoder in
            encoder.setComputePipelineState(matmulPipeline)
            // These bindings line up with [[buffer(0...3)]] in matmul_naive.
            encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
            encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer, offset: 0, index: 2)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 3)

            // Launch a 2D grid over C so each thread owns one output element.
            let width = max(1, min(matmulPipeline.threadExecutionWidth, shape.n))
            let height = max(1, min(matmulPipeline.maxTotalThreadsPerThreadgroup / width, shape.m))
            let threadsPerGroup = MTLSize(width: width, height: height, depth: 1)
            let threadsPerGrid = MTLSize(width: shape.n, height: shape.m, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        }

        let count = shape.m * shape.n
        let pointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func matmulTiled(shape: MatMulShape, lhs: [Float], rhs: [Float], config: TiledKernelConfig) throws -> [Float] {
        precondition(lhs.count == shape.m * shape.k, "lhs size must match MxK.")
        precondition(rhs.count == shape.k * shape.n, "rhs size must match KxN.")

        guard let pipeline = tiledMatmulPipelines[config] else {
            throw MetalHelloError.functionMissing(config.kernelName)
        }

        let lhsBuffer = try makeBuffer(from: lhs, label: "tiled matmul lhs")
        let rhsBuffer = try makeBuffer(from: rhs, label: "tiled matmul rhs")
        let resultBuffer = try makeEmptyBuffer(
            length: shape.m * shape.n * MemoryLayout<Float>.stride,
            label: "tiled matmul result"
        )
        let params = MatMulParams(M: UInt32(shape.m), N: UInt32(shape.n), K: UInt32(shape.k))
        let paramsBuffer = try makeBuffer(from: [params], label: "tiled matmul params")

        try runCommand(label: "tiled matmul \(config) \(shape)") { encoder in
            self.encodeTiledMatmul(
                encoder: encoder,
                pipeline: pipeline,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                resultBuffer: resultBuffer,
                paramsBuffer: paramsBuffer,
                shape: shape,
                config: config
            )
        }

        let count = shape.m * shape.n
        let pointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func matmulBiasReluTiled(shape: MatMulShape, lhs: [Float], rhs: [Float], bias: [Float], config: TiledKernelConfig) throws -> [Float] {
        precondition(lhs.count == shape.m * shape.k, "lhs size must match MxK.")
        precondition(rhs.count == shape.k * shape.n, "rhs size must match KxN.")
        precondition(bias.count == shape.n, "bias size must match N.")

        guard let pipeline = fusedTiledMatmulPipelines[config] else {
            throw MetalHelloError.functionMissing(config.fusedKernelName)
        }

        let lhsBuffer = try makeBuffer(from: lhs, label: "fused lhs")
        let rhsBuffer = try makeBuffer(from: rhs, label: "fused rhs")
        let resultBuffer = try makeEmptyBuffer(
            length: shape.m * shape.n * MemoryLayout<Float>.stride,
            label: "fused result"
        )
        let params = MatMulParams(M: UInt32(shape.m), N: UInt32(shape.n), K: UInt32(shape.k))
        let paramsBuffer = try makeBuffer(from: [params], label: "fused params")
        let biasBuffer = try makeBuffer(from: bias, label: "fused bias")

        try runCommand(label: "fused tiled matmul+bias+relu \(config) \(shape)") { encoder in
            self.encodeFusedTiledMatmulBiasRelu(
                encoder: encoder,
                pipeline: pipeline,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                resultBuffer: resultBuffer,
                paramsBuffer: paramsBuffer,
                biasBuffer: biasBuffer,
                shape: shape,
                config: config
            )
        }

        let count = shape.m * shape.n
        let pointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    func benchmarkMetalMatmul(shape: MatMulShape, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)

        for _ in 0..<warmup {
            _ = try matmul(shape: shape, lhs: lhs, rhs: rhs)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try matmul(shape: shape, lhs: lhs, rhs: rhs)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkMetalMatmul(fixture: MatMulBenchmarkFixture, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        for _ in 0..<warmup {
            _ = try matmul(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try matmul(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkTiledMetalMatmul(shape: MatMulShape, config: TiledKernelConfig, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)

        for _ in 0..<warmup {
            _ = try matmulTiled(shape: shape, lhs: lhs, rhs: rhs, config: config)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try matmulTiled(shape: shape, lhs: lhs, rhs: rhs, config: config)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkTiledMetalMatmul(fixture: MatMulBenchmarkFixture, config: TiledKernelConfig, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        for _ in 0..<warmup {
            _ = try matmulTiled(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs, config: config)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try matmulTiled(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs, config: config)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkFusedTiledMatmulBiasRelu(shape: MatMulShape, config: TiledKernelConfig, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)
        let bias = Self.randomArray(count: shape.n)

        for _ in 0..<warmup {
            _ = try matmulBiasReluTiled(shape: shape, lhs: lhs, rhs: rhs, bias: bias, config: config)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try matmulBiasReluTiled(shape: shape, lhs: lhs, rhs: rhs, bias: bias, config: config)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkFusedTiledMatmulBiasRelu(fixture: FusedBenchmarkFixture, config: TiledKernelConfig, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        for _ in 0..<warmup {
            _ = try matmulBiasReluTiled(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs, bias: fixture.bias, config: config)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try matmulBiasReluTiled(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs, bias: fixture.bias, config: config)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkComposedTiledBiasRelu(shape: MatMulShape, config: TiledKernelConfig, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)
        let bias = Self.randomArray(count: shape.n)

        let lhsBuffer = try makeBuffer(from: lhs, label: "composed lhs")
        let rhsBuffer = try makeBuffer(from: rhs, label: "composed rhs")
        let biasBuffer = try makeBuffer(from: bias, label: "composed bias")
        let resultBuffer = try makeEmptyBuffer(length: shape.m * shape.n * MemoryLayout<Float>.stride, label: "composed result")
        let matmulParams = try makeBuffer(from: [MatMulParams(M: UInt32(shape.m), N: UInt32(shape.n), K: UInt32(shape.k))], label: "composed matmul params")
        let epilogueParams = try makeBuffer(from: [EpilogueParams(M: UInt32(shape.m), N: UInt32(shape.n))], label: "composed epilogue params")

        guard let tiledPipeline = tiledMatmulPipelines[config] else {
            throw MetalHelloError.functionMissing(config.kernelName)
        }

        for _ in 0..<warmup {
            try runComposedTiledBiasRelu(
                shape: shape,
                config: config,
                tiledPipeline: tiledPipeline,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                matmulParamsBuffer: matmulParams,
                epilogueParamsBuffer: epilogueParams
            )
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            try runComposedTiledBiasRelu(
                shape: shape,
                config: config,
                tiledPipeline: tiledPipeline,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                matmulParamsBuffer: matmulParams,
                epilogueParamsBuffer: epilogueParams
            )
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkComposedTiledBiasRelu(fixture: FusedBenchmarkFixture, config: TiledKernelConfig, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        let lhsBuffer = try makeBuffer(from: fixture.lhs, label: "composed lhs")
        let rhsBuffer = try makeBuffer(from: fixture.rhs, label: "composed rhs")
        let biasBuffer = try makeBuffer(from: fixture.bias, label: "composed bias")
        let resultBuffer = try makeEmptyBuffer(length: fixture.shape.m * fixture.shape.n * MemoryLayout<Float>.stride, label: "composed result")
        let matmulParams = try makeBuffer(from: [MatMulParams(M: UInt32(fixture.shape.m), N: UInt32(fixture.shape.n), K: UInt32(fixture.shape.k))], label: "composed matmul params")
        let epilogueParams = try makeBuffer(from: [EpilogueParams(M: UInt32(fixture.shape.m), N: UInt32(fixture.shape.n))], label: "composed epilogue params")

        guard let tiledPipeline = tiledMatmulPipelines[config] else {
            throw MetalHelloError.functionMissing(config.kernelName)
        }

        for _ in 0..<warmup {
            try runComposedTiledBiasRelu(
                shape: fixture.shape,
                config: config,
                tiledPipeline: tiledPipeline,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                matmulParamsBuffer: matmulParams,
                epilogueParamsBuffer: epilogueParams
            )
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            try runComposedTiledBiasRelu(
                shape: fixture.shape,
                config: config,
                tiledPipeline: tiledPipeline,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                matmulParamsBuffer: matmulParams,
                epilogueParamsBuffer: epilogueParams
            )
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkMPSBiasRelu(shape: MatMulShape, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        guard supportsMPS else {
            throw MetalHelloError.commandEncodingFailed("MPS is not supported on device \(device.name).")
        }

        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)
        let bias = Self.randomArray(count: shape.n)

        let lhsBuffer = try makeBuffer(from: lhs, label: "mps fused lhs")
        let rhsBuffer = try makeBuffer(from: rhs, label: "mps fused rhs")
        let biasBuffer = try makeBuffer(from: bias, label: "mps fused bias")
        let resultBuffer = try makeEmptyBuffer(length: shape.m * shape.n * MemoryLayout<Float>.stride, label: "mps fused result")
        let epilogueParams = try makeBuffer(from: [EpilogueParams(M: UInt32(shape.m), N: UInt32(shape.n))], label: "mps fused params")

        for _ in 0..<warmup {
            try runMPSBiasRelu(
                shape: shape,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                epilogueParamsBuffer: epilogueParams
            )
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            try runMPSBiasRelu(
                shape: shape,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                epilogueParamsBuffer: epilogueParams
            )
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkMPSBiasRelu(fixture: FusedBenchmarkFixture, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        guard supportsMPS else {
            throw MetalHelloError.commandEncodingFailed("MPS is not supported on device \(device.name).")
        }

        let lhsBuffer = try makeBuffer(from: fixture.lhs, label: "mps fused lhs")
        let rhsBuffer = try makeBuffer(from: fixture.rhs, label: "mps fused rhs")
        let biasBuffer = try makeBuffer(from: fixture.bias, label: "mps fused bias")
        let resultBuffer = try makeEmptyBuffer(length: fixture.shape.m * fixture.shape.n * MemoryLayout<Float>.stride, label: "mps fused result")
        let epilogueParams = try makeBuffer(from: [EpilogueParams(M: UInt32(fixture.shape.m), N: UInt32(fixture.shape.n))], label: "mps fused params")

        for _ in 0..<warmup {
            try runMPSBiasRelu(
                shape: fixture.shape,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                epilogueParamsBuffer: epilogueParams
            )
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            try runMPSBiasRelu(
                shape: fixture.shape,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                biasBuffer: biasBuffer,
                resultBuffer: resultBuffer,
                epilogueParamsBuffer: epilogueParams
            )
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkMPSMatmul(shape: MatMulShape, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        guard supportsMPS else {
            throw MetalHelloError.commandEncodingFailed("MPS is not supported on device \(device.name).")
        }

        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)

        for _ in 0..<warmup {
            try runMPSMatmul(shape: shape, lhs: lhs, rhs: rhs)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            try runMPSMatmul(shape: shape, lhs: lhs, rhs: rhs)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkMPSMatmul(fixture: MatMulBenchmarkFixture, iterations: Int, warmup: Int) throws -> BenchmarkResult {
        guard supportsMPS else {
            throw MetalHelloError.commandEncodingFailed("MPS is not supported on device \(device.name).")
        }

        for _ in 0..<warmup {
            try runMPSMatmul(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            try runMPSMatmul(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkCPUMatmul(shape: MatMulShape, iterations: Int, warmup: Int) -> BenchmarkResult {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)

        for _ in 0..<warmup {
            _ = Self.cpuMatmul(shape: shape, lhs: lhs, rhs: rhs)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = Self.cpuMatmul(shape: shape, lhs: lhs, rhs: rhs)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func benchmarkCPUMatmul(fixture: MatMulBenchmarkFixture, iterations: Int, warmup: Int) -> BenchmarkResult {
        for _ in 0..<warmup {
            _ = Self.cpuMatmul(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let started = DispatchTime.now().uptimeNanoseconds
            _ = Self.cpuMatmul(shape: fixture.shape, lhs: fixture.lhs, rhs: fixture.rhs)
            let ended = DispatchTime.now().uptimeNanoseconds
            samples.append(Double(ended - started) / 1_000_000.0)
        }

        return Self.makeBenchmarkResult(samples: samples)
    }

    func verifyMatmul(shape: MatMulShape) throws -> Bool {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)
        let gpu = try matmul(shape: shape, lhs: lhs, rhs: rhs)
        let cpu = Self.cpuMatmul(shape: shape, lhs: lhs, rhs: rhs)
        return Self.allClose(gpu, cpu, tolerance: 1e-4)
    }

    func verifyTiledMatmul(shape: MatMulShape, config: TiledKernelConfig) throws -> Bool {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)
        let gpu = try matmulTiled(shape: shape, lhs: lhs, rhs: rhs, config: config)
        let cpu = Self.cpuMatmul(shape: shape, lhs: lhs, rhs: rhs)
        return Self.allClose(gpu, cpu, tolerance: 1e-4)
    }

    func verifyFusedTiledMatmulBiasRelu(shape: MatMulShape, config: TiledKernelConfig) throws -> Bool {
        let lhs = Self.randomArray(count: shape.m * shape.k)
        let rhs = Self.randomArray(count: shape.k * shape.n)
        let bias = Self.randomArray(count: shape.n)
        let gpu = try matmulBiasReluTiled(shape: shape, lhs: lhs, rhs: rhs, bias: bias, config: config)
        let cpu = Self.cpuMatmulBiasRelu(shape: shape, lhs: lhs, rhs: rhs, bias: bias)
        return Self.allClose(gpu, cpu, tolerance: 1e-4)
    }

    private func runCommand(label: String, encode: (MTLComputeCommandEncoder) throws -> Void) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalHelloError.commandEncodingFailed("Failed to create command objects for \(label).")
        }

        try encode(encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw MetalHelloError.commandEncodingFailed("\(label) failed: \(error.localizedDescription)")
        }
    }

    private func runCommandBuffer(label: String, encode: (MTLCommandBuffer) throws -> Void) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalHelloError.commandEncodingFailed("Failed to create command buffer for \(label).")
        }

        try encode(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw MetalHelloError.commandEncodingFailed("\(label) failed: \(error.localizedDescription)")
        }
    }

    private func runMPSMatmul(shape: MatMulShape, lhs: [Float], rhs: [Float]) throws {
        let lhsBuffer = try makeBuffer(from: lhs, label: "mps lhs")
        let rhsBuffer = try makeBuffer(from: rhs, label: "mps rhs")
        let resultBuffer = try makeEmptyBuffer(
            length: shape.m * shape.n * MemoryLayout<Float>.stride,
            label: "mps result"
        )

        // MPS needs explicit matrix descriptors so it can interpret the raw buffers.
        let lhsDescriptor = MPSMatrixDescriptor(
            rows: shape.m,
            columns: shape.k,
            rowBytes: shape.k * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let rhsDescriptor = MPSMatrixDescriptor(
            rows: shape.k,
            columns: shape.n,
            rowBytes: shape.n * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let resultDescriptor = MPSMatrixDescriptor(
            rows: shape.m,
            columns: shape.n,
            rowBytes: shape.n * MemoryLayout<Float>.stride,
            dataType: .float32
        )

        let lhsMatrix = MPSMatrix(buffer: lhsBuffer, descriptor: lhsDescriptor)
        let rhsMatrix = MPSMatrix(buffer: rhsBuffer, descriptor: rhsDescriptor)
        let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultDescriptor)

        // This is the tuned baseline we want to learn from and eventually challenge.
        let kernel = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: shape.m,
            resultColumns: shape.n,
            interiorColumns: shape.k,
            alpha: 1.0,
            beta: 0.0
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalHelloError.commandEncodingFailed("Failed to create command buffer for MPS matmul.")
        }

        kernel.encode(
            commandBuffer: commandBuffer,
            leftMatrix: lhsMatrix,
            rightMatrix: rhsMatrix,
            resultMatrix: resultMatrix
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw MetalHelloError.commandEncodingFailed("MPS matmul failed: \(error.localizedDescription)")
        }
    }

    private func runMPSBiasRelu(
        shape: MatMulShape,
        lhsBuffer: MTLBuffer,
        rhsBuffer: MTLBuffer,
        biasBuffer: MTLBuffer,
        resultBuffer: MTLBuffer,
        epilogueParamsBuffer: MTLBuffer
    ) throws {
        let lhsDescriptor = MPSMatrixDescriptor(
            rows: shape.m,
            columns: shape.k,
            rowBytes: shape.k * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let rhsDescriptor = MPSMatrixDescriptor(
            rows: shape.k,
            columns: shape.n,
            rowBytes: shape.n * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let resultDescriptor = MPSMatrixDescriptor(
            rows: shape.m,
            columns: shape.n,
            rowBytes: shape.n * MemoryLayout<Float>.stride,
            dataType: .float32
        )

        let lhsMatrix = MPSMatrix(buffer: lhsBuffer, descriptor: lhsDescriptor)
        let rhsMatrix = MPSMatrix(buffer: rhsBuffer, descriptor: rhsDescriptor)
        let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultDescriptor)
        let kernel = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: shape.m,
            resultColumns: shape.n,
            interiorColumns: shape.k,
            alpha: 1.0,
            beta: 0.0
        )

        try runCommandBuffer(label: "mps+bias+relu \(shape)") { commandBuffer in
            kernel.encode(
                commandBuffer: commandBuffer,
                leftMatrix: lhsMatrix,
                rightMatrix: rhsMatrix,
                resultMatrix: resultMatrix
            )
            // This is the vendor-matmul baseline plus the same GPU epilogue we use elsewhere.
            try self.encodeAddBiasReluPass(
                commandBuffer: commandBuffer,
                matrixBuffer: resultBuffer,
                biasBuffer: biasBuffer,
                paramsBuffer: epilogueParamsBuffer,
                shape: shape
            )
        }
    }

    private func runComposedTiledBiasRelu(
        shape: MatMulShape,
        config: TiledKernelConfig,
        tiledPipeline: MTLComputePipelineState,
        lhsBuffer: MTLBuffer,
        rhsBuffer: MTLBuffer,
        biasBuffer: MTLBuffer,
        resultBuffer: MTLBuffer,
        matmulParamsBuffer: MTLBuffer,
        epilogueParamsBuffer: MTLBuffer
    ) throws {
        try runCommandBuffer(label: "composed tiled+bias+relu \(config) \(shape)") { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalHelloError.commandEncodingFailed("Failed to create encoder for composed tiled path.")
            }
            // First pass: matmul only.
            self.encodeTiledMatmul(
                encoder: encoder,
                pipeline: tiledPipeline,
                lhsBuffer: lhsBuffer,
                rhsBuffer: rhsBuffer,
                resultBuffer: resultBuffer,
                paramsBuffer: matmulParamsBuffer,
                shape: shape,
                config: config
            )
            encoder.endEncoding()

            // Second pass: add bias and apply ReLU.
            try self.encodeAddBiasReluPass(
                commandBuffer: commandBuffer,
                matrixBuffer: resultBuffer,
                biasBuffer: biasBuffer,
                paramsBuffer: epilogueParamsBuffer,
                shape: shape
            )
        }
    }

    private func encodeTiledMatmul(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        lhsBuffer: MTLBuffer,
        rhsBuffer: MTLBuffer,
        resultBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        shape: MatMulShape,
        config: TiledKernelConfig
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
        encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 3)

        let threadsPerGroup = MTLSize(width: config.threadsX, height: config.threadsY, depth: 1)
        let groupsWide = (shape.n + config.outputTileN - 1) / config.outputTileN
        let groupsHigh = (shape.m + config.outputTileM - 1) / config.outputTileM
        let threadgroups = MTLSize(width: groupsWide, height: groupsHigh, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
    }

    private func encodeFusedTiledMatmulBiasRelu(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        lhsBuffer: MTLBuffer,
        rhsBuffer: MTLBuffer,
        resultBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        biasBuffer: MTLBuffer,
        shape: MatMulShape,
        config: TiledKernelConfig
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(lhsBuffer, offset: 0, index: 0)
        encoder.setBuffer(rhsBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 3)
        encoder.setBuffer(biasBuffer, offset: 0, index: 4)

        let threadsPerGroup = MTLSize(width: config.threadsX, height: config.threadsY, depth: 1)
        let groupsWide = (shape.n + config.outputTileN - 1) / config.outputTileN
        let groupsHigh = (shape.m + config.outputTileM - 1) / config.outputTileM
        let threadgroups = MTLSize(width: groupsWide, height: groupsHigh, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
    }

    private func encodeAddBiasReluPass(
        commandBuffer: MTLCommandBuffer,
        matrixBuffer: MTLBuffer,
        biasBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        shape: MatMulShape
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalHelloError.commandEncodingFailed("Failed to create encoder for bias+relu pass.")
        }
        encoder.setComputePipelineState(addBiasReluPipeline)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        encoder.setBuffer(biasBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

        let width = max(1, min(addBiasReluPipeline.threadExecutionWidth, shape.n))
        let height = max(1, min(addBiasReluPipeline.maxTotalThreadsPerThreadgroup / width, shape.m))
        let threadsPerGroup = MTLSize(width: width, height: height, depth: 1)
        let threadsPerGrid = MTLSize(width: shape.n, height: shape.m, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
    }

    private func makeBuffer<T>(from values: [T], label: String) throws -> MTLBuffer {
        let length = values.count * MemoryLayout<T>.stride
        let buffer = values.withUnsafeBytes { rawBuffer -> MTLBuffer? in
            guard let baseAddress = rawBuffer.baseAddress else {
                return device.makeBuffer(length: 0, options: .storageModeShared)
            }
            return device.makeBuffer(bytes: baseAddress, length: length, options: .storageModeShared)
        }
        guard let buffer else {
            throw MetalHelloError.bufferCreationFailed(label)
        }
        return buffer
    }

    private func makeEmptyBuffer(length: Int, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw MetalHelloError.bufferCreationFailed(label)
        }
        return buffer
    }

    private static func makeLibrary(device: MTLDevice, includeSIMDGroupSource: Bool) throws -> LibraryBuild {
        guard let shaderURL = Bundle.module.url(forResource: "ComputeKernels", withExtension: "metal") else {
            throw MetalHelloError.resourceMissing("ComputeKernels.metal")
        }

        let baseSource = try String(contentsOf: shaderURL, encoding: .utf8)

        if includeSIMDGroupSource,
           let simdSourceURL = Bundle.module.url(forResource: "SimdgroupMatrixKernels", withExtension: "metal") {
            let simdSource = try String(contentsOf: simdSourceURL, encoding: .utf8)
            do {
                let library = try device.makeLibrary(source: baseSource + "\n\n" + simdSource, options: nil)
                return LibraryBuild(library: library, includedSIMDGroupSource: true)
            } catch {
                // Fall back to the scalar-only library so unsupported toolchains do not break the build.
            }
        }

        do {
            let library = try device.makeLibrary(source: baseSource, options: nil)
            return LibraryBuild(library: library, includedSIMDGroupSource: false)
        } catch {
            throw MetalHelloError.pipelineCreationFailed(error.localizedDescription)
        }
    }

    private static func prefersSIMDGroupMatrixSource(on device: MTLDevice) -> Bool {
        if #available(macOS 12.0, *) {
            return device.supportsFamily(.apple7) || device.supportsFamily(.apple8)
        }
        return false
    }

    private static func makePipeline(named functionName: String, in library: MTLLibrary, device: MTLDevice) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalHelloError.functionMissing(functionName)
        }

        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalHelloError.pipelineCreationFailed(error.localizedDescription)
        }
    }

    private static func makeOptionalPipeline(named functionName: String, in library: MTLLibrary, device: MTLDevice) throws -> MTLComputePipelineState? {
        guard library.makeFunction(name: functionName) != nil else {
            return nil
        }
        return try makePipeline(named: functionName, in: library, device: device)
    }

    private static func randomArray(count: Int) -> [Float] {
        (0..<count).map { _ in Float.random(in: -1.0...1.0) }
    }

    private static func seededArray(count: Int, using generator: inout FixtureSeededGenerator) -> [Float] {
        (0..<count).map { _ in generator.nextFloat(in: -1.0...1.0) }
    }

    private static func makeBenchmarkResult(samples: [Double]) -> BenchmarkResult {
        precondition(!samples.isEmpty, "Benchmark must record at least one sample.")

        let sorted = samples.sorted()
        let average = samples.reduce(0, +) / Double(samples.count)
        let median: Double
        if sorted.count.isMultiple(of: 2) {
            let upper = sorted.count / 2
            median = (sorted[upper - 1] + sorted[upper]) / 2.0
        } else {
            median = sorted[sorted.count / 2]
        }

        return BenchmarkResult(
            averageMilliseconds: average,
            medianMilliseconds: median,
            minMilliseconds: sorted.first ?? average,
            maxMilliseconds: sorted.last ?? average,
            samples: samples
        )
    }

    static func cpuMatmul(shape: MatMulShape, lhs: [Float], rhs: [Float]) -> [Float] {
        var output = Array(repeating: Float.zero, count: shape.m * shape.n)
        for row in 0..<shape.m {
            for col in 0..<shape.n {
                var sum: Float = 0
                for inner in 0..<shape.k {
                    sum += lhs[row * shape.k + inner] * rhs[inner * shape.n + col]
                }
                output[row * shape.n + col] = sum
            }
        }
        return output
    }

    static func cpuMatmulBiasRelu(shape: MatMulShape, lhs: [Float], rhs: [Float], bias: [Float]) -> [Float] {
        precondition(bias.count == shape.n, "bias size must match N.")
        var output = cpuMatmul(shape: shape, lhs: lhs, rhs: rhs)
        for row in 0..<shape.m {
            for col in 0..<shape.n {
                let index = row * shape.n + col
                output[index] = max(output[index] + bias[col], 0)
            }
        }
        return output
    }

    private static func allClose(_ lhs: [Float], _ rhs: [Float], tolerance: Float) -> Bool {
        guard lhs.count == rhs.count else {
            return false
        }

        for (left, right) in zip(lhs, rhs) {
            let scale = max(1.0 as Float, max(abs(left), abs(right)))
            if abs(left - right) > tolerance * scale {
                return false
            }
        }
        return true
    }
}

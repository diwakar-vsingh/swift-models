import TensorFlow

public protocol WarmupSchedule {
  var steps: Int { get set }
  var endLearningRate: Float { get set }
  func callAsFunction(forStep step: Int) -> Float
}

public protocol DecayWithOptionalWarmupSchedule {
  var warmupSchedule: WarmupSchedule? { get set }
  var startStep: Int { get set }
  var startLearningRate: Float { get set }
  var endLearningRate: Float { get set }
  func callAsFunction(withEndStep endStep: Int, forStep step: Int) -> Float
}

public struct LinearWarmupSchedule: WarmupSchedule {
  public var steps: Int
  public var endLearningRate: Float

  public init(steps: Int, endLearningRate: Float) {
    self.steps = steps
    self.endLearningRate = endLearningRate
  }

  public func callAsFunction(forStep step: Int) -> Float {
    return LinearLearningRateCurve(
      startStep: 0,
      endStep: steps,
      startLearningRate: 0,
      endLearningRate: endLearningRate)(atStep: step)
  }
}

public struct LinearDecayWithOptionalWarmupSchedule: DecayWithOptionalWarmupSchedule {
  public var warmupSchedule: WarmupSchedule?
  public var startStep: Int
  public var startLearningRate: Float
  public var endLearningRate: Float

  public init(
    warmupSchedule: WarmupSchedule? = nil,
    startLearningRate: Float? = nil,
    endLearningRate: Float = 0
  ) {
    if let warmupSchedule = warmupSchedule {
      precondition(
        startLearningRate == nil,
        "Shouldn't specify startLearningRate when warmupSchedule is provided.")
      self.startStep = warmupSchedule.steps
      self.startLearningRate = warmupSchedule.endLearningRate
    } else {
      precondition(
        startLearningRate != nil,
        "Should specify startLearningRate when warmupSchedule is nil.")
      self.startStep = 0
      self.startLearningRate = startLearningRate!
    }
    self.endLearningRate = endLearningRate
  }

  public func callAsFunction(withEndStep endStep: Int, forStep step: Int) -> Float {
    if let warmupSchedule = warmupSchedule {
      if step <= warmupSchedule.steps {
        return warmupSchedule(forStep: step)
      }
    }
    return LinearLearningRateCurve(
      startStep: startStep,
      endStep: endStep,
      startLearningRate: startLearningRate,
      endLearningRate: endLearningRate)(atStep: step)

  }
}

public func learningRateScheduler<L: TrainingLoopProtocol>(
  schedule: DecayWithOptionalWarmupSchedule,
  biasCorrectionBeta: (Float, Float)? = nil
) -> TrainingLoopCallback<L> {

  var totalStepCount: Int = 0

  return { (loop, event) throws -> Void in
    if event != .batchStart || Context.local.learningPhase == .inference { return }

    if totalStepCount == 0 {
      totalStepCount = loop.batchCount! * loop.epochCount!
    }

    let step = loop.batchIndex! + loop.epochIndex! * loop.batchCount! + 1

    var learningRate = schedule(withEndStep: totalStepCount, forStep: step)
    if let beta = biasCorrectionBeta {
      learningRate *= sqrtf(1 - powf(beta.1, Float(step))) / (beta.0 - powf(beta.1, Float(step)))
    }

    loop.optimizer.learningRate = learningRate as! L.Opt.Scalar
  }
}

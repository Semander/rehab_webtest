import Foundation
import MediaPipeTasksVision

final class PoseEstimator: NSObject {

    // in live mode the result comes back later through the delegate so we hand
    // it out with this closure
    var onLiveResult: ((Pose) -> Void)?

    private var landmarker: PoseLandmarker!
    private let runningMode: RunningMode

    init(modelType: ModelType = .full,
         runningMode: RunningMode = .liveStream) throws {
        guard let modelPath = modelType.modelPath else {
            throw PoseEstimatorError.modelNotFound(modelType)
        }
        self.runningMode = runningMode
        super.init()

        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = runningMode
        options.numPoses = 1

        if runningMode == .liveStream {
            options.poseLandmarkerLiveStreamDelegate = self
        }
        self.landmarker = try PoseLandmarker(options: options)
    }

    func detectAsync(sampleBuffer: CMSampleBuffer,
                     orientation: UIImage.Orientation,
                     timestampMs: Int) {
        guard runningMode == .liveStream else { return }

        guard let image = try? MPImage(sampleBuffer: sampleBuffer, orientation: orientation) else {
            return
        }
        try? landmarker.detectAsync(image: image, timestampInMilliseconds: timestampMs)
    }

    func detect(videoFrame sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation = .up, timestampMs: Int) -> Pose? {
        guard runningMode == .video else { return nil }
        guard let image = try? MPImage(sampleBuffer: sampleBuffer, orientation: orientation),
              let result = try? landmarker.detect(videoFrame: image, timestampInMilliseconds: timestampMs)
        else {
            return nil
        }
        return Pose(result: result)
    }

    func close() {
        landmarker = nil
    }
}

extension PoseEstimator: PoseLandmarkerLiveStreamDelegate {
    func poseLandmarker(_ poseLandmarker: PoseLandmarker,
                        didFinishDetection result: PoseLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        guard let result else { return }
        onLiveResult?(Pose(result: result))
    }
}

enum PoseEstimatorError: LocalizedError {
    case modelNotFound(ModelType)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let type):
            return "Model \(type.rawValue).task not found in the app bundle. " +
                   "Add pose_landmarker_\(type.rawValue).task to the target's resources."
        }
    }
}

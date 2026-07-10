import Foundation
import AVFoundation
import CoreImage

final class VideoPoseRunner: NSObject, ObservableObject, PoseSource {

    @Published var latestPose: Pose?
    @Published var imageSize: CGSize = .zero
    @Published var errorMessage: String?
    @Published var displayImage: CGImage?

    private let ciContext = CIContext()
    private let modelType: ModelType
    private let workQueue = DispatchQueue(label: "rehab.video.runner")
    private var isRunning = false

    init(modelType: ModelType = .full) {
        self.modelType = modelType
        super.init()
    }

    func start() {}

    func stop() {
        isRunning = false
    }

    func play(url: URL) {
        stop()
        workQueue.async { [weak self] in
            guard let self else { return }
            DispatchQueue.main.async {
                self.errorMessage = nil
                self.imageSize = .zero
                self.displayImage = nil
            }
            guard let extractor = VideoFrameExtractor(url: url) else {
                DispatchQueue.main.async { self.errorMessage = "Could not read the selected video." }
                return
            }
            let estimator: PoseEstimator
            do {
                estimator = try PoseEstimator(modelType: self.modelType, runningMode: .video)
            } catch {
                DispatchQueue.main.async { self.errorMessage = error.localizedDescription }
                return
            }
            self.isRunning = true
            self.playLoop(extractor: extractor, estimator: estimator)
        }
    }

    private func playLoop(extractor: VideoFrameExtractor, estimator: PoseEstimator) {
        // MediaPipe vide mode wants timestamps that always go up. but every time
        // we loop the clip the timestamps restart at 0, which it hates; so keep
        // adding this base offset each loop so they keep climbing.
        var timestampBase = 0
        let frameIntervalMs = Int(1000.0 / Double(max(extractor.nominalFrameRate, 1)))

        while isRunning {
            extractor.rewind()
            var previousMs: Int?
            var lastClipMs = 0
            while isRunning, let frame = extractor.nextFrame() {
                if let prev = previousMs {
                    let waitMs = frame.timestampMs - prev
                    if waitMs > 0 { Thread.sleep(forTimeInterval: Double(waitMs) / 1000.0) }
                }
                previousMs = frame.timestampMs
                lastClipMs = frame.timestampMs

                let pose = estimator.detect(videoFrame: frame.sampleBuffer,
                                            timestampMs: timestampBase + frame.timestampMs)
                let image = renderFrame(frame.sampleBuffer)
                DispatchQueue.main.async {
                    self.latestPose = pose
                    if let image { self.displayImage = image }
                }
            }
            timestampBase += lastClipMs + frameIntervalMs
        }
    }

    private func renderFrame(_ sampleBuffer: CMSampleBuffer) -> CGImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        if imageSize == .zero {
            let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                              height: CVPixelBufferGetHeight(pixelBuffer))
            DispatchQueue.main.async { self.imageSize = size }
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        return ciContext.createCGImage(ciImage, from: ciImage.extent)
    }
}

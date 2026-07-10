import Foundation
import AVFoundation
import UIKit

final class CameraManager: NSObject, ObservableObject, PoseSource {
    @Published var latestPose: Pose?
    @Published var imageSize: CGSize = .zero
    @Published var errorMessage: String?

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "rehab.camera.session")
    private let videoQueue = DispatchQueue(label: "rehab.camera.video")
    private var estimator: PoseEstimator?
    private let modelType: ModelType

    init(modelType: ModelType = .full) {
        self.modelType = modelType
        super.init()
    }

    func start() {
        requestAccessIfNeeded { [weak self] granted in
            guard let self else { return }
            guard granted else {
                DispatchQueue.main.async { self.errorMessage = "Camera access denied." }
                return
            }
            self.sessionQueue.async {
                self.configureIfNeeded()
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    // *** setup ***
    private var isConfigured = false

    private func configureIfNeeded() {
        guard !isConfigured else { return }

        do {
            estimator = try PoseEstimator(modelType: modelType, runningMode: .liveStream)
            estimator?.onLiveResult = { [weak self] pose in
                DispatchQueue.main.async { self?.latestPose = pose }
            }
        } catch {
            DispatchQueue.main.async { self.errorMessage = error.localizedDescription }
            return
        }

        session.beginConfiguration()
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.errorMessage = "No usable camera found." }
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true

        // the camera gives YUV frames by default but MediaPipe only accepts BGRA
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }

        session.commitConfiguration()
        isConfigured = true
    }

    private func requestAccessIfNeeded(_ completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { completion($0) }
        default:
            completion(false)
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if imageSize == .zero, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let size = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
            DispatchQueue.main.async { self.imageSize = size }
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestampMs = Int(CMTimeGetSeconds(pts) * 1000)
        estimator?.detectAsync(sampleBuffer: sampleBuffer, orientation: .up, timestampMs: timestampMs)
    }
}

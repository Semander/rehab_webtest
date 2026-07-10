import Foundation
import AVFoundation
import CoreMedia

final class VideoFrameExtractor {

    struct Frame {
        let sampleBuffer: CMSampleBuffer
        let timestampMs: Int
    }

    let naturalSize: CGSize
    let nominalFrameRate: Float

    private let asset: AVAsset
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private let videoTrack: AVAssetTrack

    init?(url: URL) {
        let asset = AVAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else { return nil }
        self.asset = asset
        self.videoTrack = track
        self.naturalSize = track.naturalSize.applying(track.preferredTransform).absoluteSize
        self.nominalFrameRate = track.nominalFrameRate > 0 ? track.nominalFrameRate : 30
    }

    func rewind() {
        reader?.cancelReading()
        guard let reader = try? AVAssetReader(asset: asset) else { return }
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { return }
        reader.add(output)
        reader.startReading()
        self.reader = reader
        self.output = output
    }

    func nextFrame() -> Frame? {
        guard let output, reader?.status == .reading,
              let sample = output.copyNextSampleBuffer() else { return nil }
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        return Frame(sampleBuffer: sample, timestampMs: Int(CMTimeGetSeconds(pts) * 1000))
    }
}
private extension CGSize {
    var absoluteSize: CGSize { CGSize(width: abs(width), height: abs(height)) }
}

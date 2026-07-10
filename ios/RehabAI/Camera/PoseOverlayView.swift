import SwiftUI
import MediaPipeTasksVision

struct PoseOverlayView: View {
    let pose: Pose?
    let imageSize: CGSize
    let contentMode: ContentMode

    var body: some View {
        Canvas { context, size in
            guard let pose, pose.isDetected else { return }
            let landmarks = pose.landmarksList[0]
            let map = mapper(viewSize: size)

            var path = Path()
            for (a, b) in PoseConnections.edges {
                guard a.rawValue < landmarks.count, b.rawValue < landmarks.count else { continue }
                path.move(to: map(landmarks[a.rawValue]))
                path.addLine(to: map(landmarks[b.rawValue]))
            }
            context.stroke(path, with: .color(.green), lineWidth: 3)

            for lm in landmarks {
                let p = map(lm)
                let dot = Path(ellipseIn: CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8))
                context.fill(dot, with: .color(.red))
            }
        }
        .allowsHitTesting(false)
    }

    private func mapper(viewSize: CGSize) -> (NormalizedLandmark) -> CGPoint {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return { CGPoint(x: CGFloat($0.x) * viewSize.width, y: CGFloat($0.y) * viewSize.height) }
        }
        let sx = viewSize.width / imageSize.width
        let sy = viewSize.height / imageSize.height
        let scale = contentMode == .fill ? max(sx, sy) : min(sx, sy)
        let displayed = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let offset = CGPoint(x: (viewSize.width - displayed.width) / 2, y: (viewSize.height - displayed.height) / 2)
        return { lm in
            CGPoint(x: offset.x + CGFloat(lm.x) * displayed.width, y: offset.y + CGFloat(lm.y) * displayed.height)
        }
    }
}

import Foundation
import MediaPipeTasksVision

struct Pose {
    let landmarksList: [[NormalizedLandmark]]
    let worldLandmarksList: [[Landmark]]

    init(result: PoseLandmarkerResult) {
        self.landmarksList = result.landmarks
        self.worldLandmarksList = result.worldLandmarks
    }

    var isDetected: Bool {
        !landmarksList.isEmpty
    }

    func getLandmark(_ poseLandmark: PoseLandmarkIndex) -> Landmark? {
        if !isDetected { return nil }
        return worldLandmarksList[0][poseLandmark.rawValue]
    }
}

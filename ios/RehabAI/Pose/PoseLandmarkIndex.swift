import Foundation

/// the 33 MediaPipe Pose landmarks
enum PoseLandmarkIndex: Int, CaseIterable {
    case nose = 0
    case leftEyeInner = 1
    case leftEye = 2
    case leftEyeOuter = 3
    case rightEyeInner = 4
    case rightEye = 5
    case rightEyeOuter = 6
    case leftEar = 7
    case rightEar = 8
    case mouthLeft = 9
    case mouthRight = 10
    case leftShoulder = 11
    case rightShoulder = 12
    case leftElbow = 13
    case rightElbow = 14
    case leftWrist = 15
    case rightWrist = 16
    case leftPinky = 17
    case rightPinky = 18
    case leftIndex = 19
    case rightIndex = 20
    case leftThumb = 21
    case rightThumb = 22
    case leftHip = 23
    case rightHip = 24
    case leftKnee = 25
    case rightKnee = 26
    case leftAnkle = 27
    case rightAnkle = 28
    case leftHeel = 29
    case rightHeel = 30
    case leftFootIndex = 31
    case rightFootIndex = 32
}

enum PoseConnections {
    static let edges: [(PoseLandmarkIndex, PoseLandmarkIndex)] = [
        // torso
        (.leftShoulder, .rightShoulder),
        (.leftShoulder, .leftHip),
        (.rightShoulder, .rightHip),
        (.leftHip, .rightHip),
        // left arm
        (.leftShoulder, .leftElbow),
        (.leftElbow, .leftWrist),
        // right arm
        (.rightShoulder, .rightElbow),
        (.rightElbow, .rightWrist),
        // left leg
        (.leftHip, .leftKnee),
        (.leftKnee, .leftAnkle),
        (.leftAnkle, .leftHeel),
        (.leftHeel, .leftFootIndex),
        (.leftAnkle, .leftFootIndex),
        // right leg
        (.rightHip, .rightKnee),
        (.rightKnee, .rightAnkle),
        (.rightAnkle, .rightHeel),
        (.rightHeel, .rightFootIndex),
        (.rightAnkle, .rightFootIndex),
    ]
}

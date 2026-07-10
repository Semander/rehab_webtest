import Foundation
import simd
import MediaPipeTasksVision

struct PoseAnalyzer {
    let pose: Pose

    init(pose: Pose) {
        self.pose = pose
    }

    var leftElbow: Double? {
        if !pose.isDetected { return nil }

        guard let shoulder = pose.getLandmark(.leftShoulder),
              let elbow = pose.getLandmark(.leftElbow),
              let wrist = pose.getLandmark(.leftWrist) else { return nil }

        let shoulderVector = SIMD3<Double>(Double(shoulder.x), Double(shoulder.y), Double(shoulder.z))
        let elbowVector = SIMD3<Double>(Double(elbow.x), Double(elbow.y), Double(elbow.z))
        let wristVector = SIMD3<Double>(Double(wrist.x), Double(wrist.y), Double(wrist.z))

        let angle = getJointAngle(elbowVector, wristVector, shoulderVector)
        return angle
    }

    var rightElbow: Double? {
        if !pose.isDetected { return nil }

        guard let shoulder = pose.getLandmark(.rightShoulder),
              let elbow = pose.getLandmark(.rightElbow),
              let wrist = pose.getLandmark(.rightWrist) else { return nil }

        let shoulderVector = SIMD3<Double>(Double(shoulder.x), Double(shoulder.y), Double(shoulder.z))
        let elbowVector = SIMD3<Double>(Double(elbow.x), Double(elbow.y), Double(elbow.z))
        let wristVector = SIMD3<Double>(Double(wrist.x), Double(wrist.y), Double(wrist.z))

        let angle = getJointAngle(elbowVector, wristVector, shoulderVector)
        return angle
    }

    var leftShoulder: Double? {
        if !pose.isDetected { return nil }

        guard let shoulder = pose.getLandmark(.leftShoulder),
              let elbow = pose.getLandmark(.leftElbow),
              let hip = pose.getLandmark(.leftHip) else { return nil }

        let shoulderVector = SIMD3<Double>(Double(shoulder.x), Double(shoulder.y), Double(shoulder.z))
        let elbowVector = SIMD3<Double>(Double(elbow.x), Double(elbow.y), Double(elbow.z))
        let hipVector = SIMD3<Double>(Double(hip.x), Double(hip.y), Double(hip.z))

        let angle = getJointAngle(shoulderVector, elbowVector, hipVector)
        return angle
    }

    var rightShoulder: Double? {
        if !pose.isDetected { return nil }

        guard let shoulder = pose.getLandmark(.rightShoulder),
              let elbow = pose.getLandmark(.rightElbow),
              let hip = pose.getLandmark(.rightHip) else { return nil }

        let shoulderVector = SIMD3<Double>(Double(shoulder.x), Double(shoulder.y), Double(shoulder.z))
        let elbowVector = SIMD3<Double>(Double(elbow.x), Double(elbow.y), Double(elbow.z))
        let hipVector = SIMD3<Double>(Double(hip.x), Double(hip.y), Double(hip.z))

        let angle = getJointAngle(shoulderVector, elbowVector, hipVector)
        return angle
    }

    private func getJointAngle(_ vertexPoint: SIMD3<Double>, _ endPoint1: SIMD3<Double>, _ endPoint2: SIMD3<Double>) -> Double {
        let ba = endPoint1 - vertexPoint
        let bc = endPoint2 - vertexPoint

        let cosineAngle = simd_dot(ba, bc) / (simd_length(ba) * simd_length(bc))

        let angle = acos(cosineAngle) * 180.0 / .pi
        return angle
    }
}

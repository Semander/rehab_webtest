import Foundation

protocol PoseSource: ObservableObject {
    var latestPose: Pose? { get }
    var imageSize: CGSize { get }
    var errorMessage: String? { get }

    func start()
    func stop()
}

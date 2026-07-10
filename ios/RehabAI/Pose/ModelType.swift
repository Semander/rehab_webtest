import Foundation

enum ModelType: String, CaseIterable {
    case lite
    case full
    case heavy

    // path to models/pretrained/...
    private var resourceName: String {
        switch self {
        case .lite:  return "pose_landmarker_lite"
        case .full:  return "pose_landmarker_full"
        case .heavy: return "pose_landmarker_heavy"
        }
    }

    var modelPath: String? {
        Bundle.main.path(forResource: resourceName, ofType: "task")
    }
}

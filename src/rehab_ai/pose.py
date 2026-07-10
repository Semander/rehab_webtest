import mediapipe as mp
import numpy as np

class Pose():
    def __init__(self, detection_result):
        self.landmarks_list = detection_result.pose_landmarks
        self.world_landmarks_list = detection_result.pose_world_landmarks

    @property
    def is_detected(self):
        return len(self.landmarks_list) > 0

    def draw_landmarks(self, frame):
        # looked up lazily so Pose objects can be built on mediapipe versions
        # that lack the drawing helpers (only needed when actually drawing)
        self._DrawingStyles = mp.tasks.vision.drawing_styles
        self._DrawingUtils = mp.tasks.vision.drawing_utils
        annotated_image = np.copy(frame)

        pose_landmark_style = self._DrawingStyles.get_default_pose_landmarks_style()
        pose_connection_style = self._DrawingUtils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for landmarks in self.landmarks_list:
            self._DrawingUtils.draw_landmarks(
                image = annotated_image,
                landmark_list = landmarks,
                connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec = pose_landmark_style,
                connection_drawing_spec = pose_connection_style
            )
        return annotated_image
    
    def get_landmark(self, pose_landmark=mp.tasks.vision.PoseLandmark):
        if not self.is_detected:
            return None
        return self.world_landmarks_list[0][pose_landmark]
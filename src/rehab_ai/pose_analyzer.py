import mediapipe as mp
import numpy as np

class PoseAnalyzer():
    def __init__(self, pose):
        self.pose = pose

    # all arm angles including joint angles that involve the shoulder, elbow, and write landmarks
    @property
    def left_elbow(self):
        if not self.pose.is_detected:
            return None
        
        shoulder = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.LEFT_SHOULDER)
        shoulder_vector = np.array([shoulder.x, shoulder.y, shoulder.z])

        elbow = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.LEFT_ELBOW)
        elbow_vector = np.array([elbow.x, elbow.y, elbow.z])

        wrist = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.LEFT_WRIST)
        wrist_vector = np.array([wrist.x, wrist.y, wrist.z])

        angle = self._get_joint_angle(elbow_vector, wrist_vector, shoulder_vector)

        return angle
    
    @property
    def right_elbow(self):
        if not self.pose.is_detected:
            return None
        
        shoulder = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER)
        shoulder_vector = np.array([shoulder.x, shoulder.y, shoulder.z])

        elbow = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.RIGHT_ELBOW)
        elbow_vector = np.array([elbow.x, elbow.y, elbow.z])

        wrist = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.RIGHT_WRIST)
        wrist_vector = np.array([wrist.x, wrist.y, wrist.z])

        angle = self._get_joint_angle(elbow_vector, wrist_vector, shoulder_vector)

        return angle
    
    @property
    def left_shoulder(self):
        if not self.pose.is_detected:
            return None
        
        shoulder = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.LEFT_SHOULDER)
        shoulder_vector = np.array([shoulder.x, shoulder.y, shoulder.z])

        elbow = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.LEFT_ELBOW)
        elbow_vector = np.array([elbow.x, elbow.y, elbow.z])

        hip = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.LEFT_HIP)
        hip_vector = np.array([hip.x, hip.y, hip.z])

        angle = self._get_joint_angle(shoulder_vector, elbow_vector, hip_vector)

        return angle
    
    @property
    def right_shoulder(self):
        if not self.pose.is_detected:
            return None
        
        shoulder = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER)
        shoulder_vector = np.array([shoulder.x, shoulder.y, shoulder.z])

        elbow = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.RIGHT_ELBOW)
        elbow_vector = np.array([elbow.x, elbow.y, elbow.z])

        hip = self.pose.get_landmark(mp.tasks.vision.PoseLandmark.RIGHT_HIP)
        hip_vector = np.array([hip.x, hip.y, hip.z])

        angle = self._get_joint_angle(shoulder_vector, elbow_vector, hip_vector)

        return angle
    
    def _get_joint_angle(self, vertex_point, end_point_1, end_point_2):
        BA = end_point_1 - vertex_point
        BC = end_point_2 - vertex_point

        cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))

        angle = np.degrees(np.arccos(cosine_angle))
        return angle
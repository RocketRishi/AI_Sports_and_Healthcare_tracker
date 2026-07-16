import mediapipe as mp

mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(static_image_mode=False)


# Map important landmarks
LANDMARK_MAP = { 
  "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
  "hip": mp_pose.PoseLandmark.LEFT_HIP,
  "knee": mp_pose.PoseLandmark.LEFT_KNEE,
  "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
  "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
  "wrist": mp_pose.PoseLandmark.LEFT_WRIST 
} 


def extract_keypoints(frames):
  all_keypoints = [] 
  
  for frame in frames: 
    results = pose.process(frame) 
    
    frame_data = {}
    
    if results.pose_landmarks:
      landmarks = results.pose_landmarks.landmark
      
      for name, idx in LANDMARK_MAP.items():
        lm = landmarks[idx]
        frame_data[name] = [lm.x, lm.y, lm.visibility]
    
    else: 
      frame_data = {} 
      
      
    all_keypoints.append(frame_data)
    
  return all_keypoints

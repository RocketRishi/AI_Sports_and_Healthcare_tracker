import numpy as np


def calculate_angle(A, B, C):
  A = np.array(A)
  B = np.array(B)
  C = np.array(C)
  
  BA = A - B
  BC = C - B 
  
  cos_angle = np.dot(BA, BC) / (
    np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6
  )
  
  angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
  
  return angle

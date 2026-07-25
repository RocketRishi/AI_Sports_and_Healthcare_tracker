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


def calculate_flexion_angle(A, B, C):
  """
  Joint flexion expressed as deviation from a fully straight limb
  (0 degrees = straight, larger = more bent), which is the convention
  used in biomechanics literature. calculate_angle() instead returns
  the raw interior angle at B (180 degrees = straight), so this is
  just that value flipped.
  """
  return 180.0 - calculate_angle(A, B, C)


def calculate_lean_angle(base, top):
  """
  Angle between the base->top segment (e.g. hip->shoulder) and true
  vertical, in degrees. 0 = upright, larger = more forward/lateral
  lean. Image coordinates are used as given (y increases downward),
  so "up" is the vector (0, -1).
  """
  base = np.array(base, dtype=float)
  top = np.array(top, dtype=float)

  segment = top - base
  vertical = np.array([0.0, -1.0])

  cos_angle = np.dot(segment, vertical) / (
    np.linalg.norm(segment) * np.linalg.norm(vertical) + 1e-6
  )

  return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

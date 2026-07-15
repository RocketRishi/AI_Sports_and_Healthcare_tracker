def is_low_confidence(keypoints):
  total_conf = 0
  count = 0
  missing = 0 
  
  for frame in keypoints:
    if not frame:
      missing += 1
      continue
      
    for kp in frame.values():
      conf = kp[2]
      total_conf += conf
      count += 1
      
      if conf < 0.2:
        missing += 1
        
  if count == 0:
    return True
    
  avg_conf = total_conf / count
  missing_ratio = missing / max(count, 1)
  
  return avg_conf < 0.6 or missing_ratio > 0.2

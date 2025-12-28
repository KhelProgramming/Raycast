import numpy as np

def preprocess_landmarks(landmarks):
    """
    Translates landmarks relative to wrist (point 0),
    normalizes by scale, and flattens to 1D array.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Translate to wrist (0,0,0) origin
    coords -= coords[0]
    
    # Scale based on distance between wrist (0) and middle finger mcp (9) or similar
    # Using original code logic: index 0 to 12 (middle finger tip approx distance metric)
    norm_factor = np.linalg.norm(coords[0] - coords[12])
    
    if norm_factor > 0:
        coords /= norm_factor
        
    return coords.flatten()

def get_hand_distance(hand_landmarks):
    """
    Calculate depth based on HAND SIZE (Wrist to Middle Finger MCP).
    Returns NEGATIVE value so that:
      - Large hand (Close) = More Negative (e.g. -0.4)
      - Small hand (Far)   = Less Negative (e.g. -0.05)
    """
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    
    # Calculate Euclidean distance on screen (X/Y only)
    # Larger distance = Closer to camera
    size = np.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
    
    # Return negative size to match existing "Lower value = Closer" logic
    return -size
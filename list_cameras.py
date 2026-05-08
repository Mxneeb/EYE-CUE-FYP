"""
List all available cameras to find the EagleEye camera index.
"""

import cv2

def list_cameras(max_test=10):
    """Test camera indices to find available cameras."""
    available_cameras = []
    
    print("Scanning for available cameras...\n")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend = cap.getBackendName()
                
                # Try to get camera name (not always available)
                print(f"Camera {i}:")
                print(f"  Resolution: {width}x{height}")
                print(f"  Backend: {backend}")
                print(f"  Status: Available\n")
                
                available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("No cameras found!")
    else:
        print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")
        print("\nTo use a specific camera, update the camera index in nav_assist/app.py")
        print("Look for: cap = cv2.VideoCapture(INDEX)")
    
    return available_cameras


if __name__ == '__main__':
    list_cameras()

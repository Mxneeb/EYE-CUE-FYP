"""
Enhanced camera detection with device names to find EagleEye camera.
"""

import cv2
import subprocess
import re

def get_camera_names_windows():
    """Get camera device names on Windows using PowerShell."""
    try:
        # Use PowerShell to list video devices
        cmd = 'powershell "Get-PnpDevice -Class Camera | Select-Object FriendlyName, Status"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("Available Camera Devices (from Windows):")
        print("=" * 60)
        print(result.stdout)
        print("=" * 60)
    except Exception as e:
        print(f"Could not get device names: {e}")

def test_all_cameras(max_test=10):
    """Test all camera indices and try to identify them."""
    print("\nTesting OpenCV Camera Indices:")
    print("=" * 60)
    
    available = []
    
    for i in range(max_test):
        # Try with DSHOW
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                print(f"\nCamera Index {i}:")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                print(f"  Backend: DSHOW")
                
                # Try to get more info
                try:
                    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                    print(f"  Codec: {codec}")
                except:
                    pass
                
                available.append(i)
            cap.release()
    
    print("\n" + "=" * 60)
    if available:
        print(f"\nFound {len(available)} camera(s) at indices: {available}")
        print("\nBased on the Windows device list above:")
        print("- Look for 'EagleEye' or similar USB camera name")
        print("- Match it to the camera index by process of elimination")
        print("\nKnown cameras:")
        print("  Index 0 = iVCam")
        print("  Index 3 = OBS Virtual Camera")
        print("  Index ? = EagleEye (check if indices 1, 2, 4, 5, etc. work)")
    else:
        print("\nNo cameras found!")
    
    return available

if __name__ == '__main__':
    print("EagleEye Camera Finder")
    print("=" * 60)
    
    # Get Windows device names
    get_camera_names_windows()
    
    # Test OpenCV indices
    available = test_all_cameras(max_test=10)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Look at the Windows device list above")
    print("2. Find the EagleEye camera name")
    print("3. Try indices that weren't 0 or 3")
    print("4. If EagleEye doesn't appear, check:")
    print("   - USB cable is connected")
    print("   - Camera is powered on")
    print("   - Windows recognizes it (Device Manager)")
    print("=" * 60)

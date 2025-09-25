import cv2

print("Testing camera...")
i = 0
print(f"Trying camera {i}...")

# Force OpenCV to use V4L2 backend (critical for exposure control on Linux)
cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"Camera {i} works! Frame shape: {frame.shape}")

        # Check if exposure property is supported
        exposure_prop = cv2.CAP_PROP_EXPOSURE
        current_exp = cap.get(exposure_prop)
        print(f"Initial exposure value: {current_exp}")

        # Try setting a valid exposure range (typically -7 to 1 for many cameras)
        # Set to manual mode first if possible
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode (V4L2)

        # Set initial exposure
        cap.set(exposure_prop, -6)
        print("Exposure set to manual mode. Initial value: -6")

        print("Press 'q' to quit, '+' to increase exposure, '-' to decrease exposure")
        print("Note: Exposure range varies by camera. Try values between -10 and 5.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Get current exposure value (may still return -1 if unsupported)
            current_exp = cap.get(exposure_prop)
            exposure_str = f"{current_exp:.1f}" if current_exp != -1 else "Unknown (unsupported)"

            # Display the frame with exposure info
            cv2.imshow(f'Camera {i} (Exposure: {exposure_str})', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('+'):
                current_exp = cap.get(exposure_prop)
                if current_exp != -1:
                    new_exp = min(current_exp + 1, 5)  # Upper limit
                    cap.set(exposure_prop, new_exp)
                    print(f"Exposure increased to: {cap.get(exposure_prop):.1f}")
                else:
                    print("Exposure control not supported or in auto mode.")
            elif key == ord('-'):
                current_exp = cap.get(exposure_prop)
                if current_exp != -1:
                    new_exp = max(current_exp - 1, -10)  # Lower limit
                    cap.set(exposure_prop, new_exp)
                    print(f"Exposure decreased to: {cap.get(exposure_prop):.1f}")
                else:
                    print("Exposure control not supported or in auto mode.")

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Camera opened but could not read frame.")
else:
    print(f"Camera {i} not available. Try using a different backend or check camera permissions.")
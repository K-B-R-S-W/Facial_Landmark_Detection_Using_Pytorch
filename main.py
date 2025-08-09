"""
Fixed Main application for HRNetV2 Facial Landmark Detection
Uses hrnetv2_w32_imagenet_pretrained.pth as backbone
"""
import os
import sys
import cv2
import torch

# Add project root and src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def init_webcam(camera_index=0):
    """Initialize webcam with error handling"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap

def get_frame(cap):
    """Get frame from webcam with error handling"""
    if cap is None:
        return None
        
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        return None
    return frame

def show_frame(frame, window_name="HRNetV2 Landmark Detection"):
    """Display frame and return key press"""
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF

def main():
    print("=" * 60)
    print("HRNetV2 Facial Landmark Detection Application (FIXED)")
    print("=" * 60)
    
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")
    
    # Model path - Update this to your actual path
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "models", 
        "hrnetv2_w32_imagenet_pretrained.pth"
    )
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model file is in the correct location")
        return
    
    try:
        # Import the fixed HRNetV2 landmark detector
        from src.landmark_detector import LandmarkDetector
        
        # Initialize detector with HRNetV2 ImageNet pretrained weights
        print(f"Loading model from: {model_path}")
        detector = LandmarkDetector(model_path=model_path, device=device)
        
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        import traceback
        traceback.print_exc()
        print("Exiting...")
        return
    
    # Initialize webcam
    cap = init_webcam()
    if cap is None:
        print("❌ Failed to initialize webcam. Exiting...")
        return
    
    print("✅ Webcam initialized successfully!")
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  'q' - Quit application")
    print("  's' - Save current frame with landmarks")
    print("  'r' - Reset frame counter")
    print("  'd' - Toggle debug mode")
    print("=" * 60)
    
    frame_count = 0
    saved_count = 0
    debug_mode = False
    detection_errors = 0
    
    try:
        while True:
            # Get frame from webcam
            frame = get_frame(cap)
            if frame is None:
                break
            
            frame_count += 1
            
            # Detect landmarks using HRNetV2 backbone
            try:
                result = detector.detect_landmarks(frame)
                if result is None:
                    detection_errors += 1
                    if debug_mode:
                        print(f"Detection returned None on frame {frame_count}")
            except Exception as e:
                detection_errors += 1
                if frame_count % 30 == 0 or debug_mode:  # Print error every 30 frames to avoid spam
                    print(f"Detection error on frame {frame_count}: {e}")
                result = None
            
            # Draw landmarks on frame
            try:
                frame = detector.draw_landmarks(frame, result)
            except Exception as e:
                if debug_mode:
                    print(f"Drawing error on frame {frame_count}: {e}")
                # Add basic error text if drawing fails
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Drawing Error - Check Console", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Add frame counter and stats to display
            cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if detection_errors > 0:
                error_rate = (detection_errors / frame_count) * 100
                cv2.putText(frame, f"Errors: {error_rate:.1f}%", (frame.shape[1] - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Show frame
            key = show_frame(frame)
            
            # Handle key presses
            if key == ord('q'):
                print("\n✅ Quit requested by user")
                break
            elif key == ord('s'):
                # Save current frame
                saved_count += 1
                filename = f"hrnetv2_landmarks_{saved_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved frame as {filename}")
            elif key == ord('r'):
                # Reset frame counter
                frame_count = 0
                detection_errors = 0
                print("🔄 Frame counter and stats reset")
            elif key == ord('d'):
                # Toggle debug mode
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                success_rate = ((frame_count - detection_errors) / frame_count) * 100
                print(f"📊 Processed {frame_count} frames... Success rate: {success_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("✅ Application closed successfully")
        print(f"📈 Total frames processed: {frame_count}")
        print(f"💾 Total frames saved: {saved_count}")
        if frame_count > 0:
            success_rate = ((frame_count - detection_errors) / frame_count) * 100
            print(f"🎯 Detection success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    main()
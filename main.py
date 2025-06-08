import cv2
import numpy as np
import pyautogui
import time
import sys
import os
import keyboard
from PIL import Image

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hand_tracker import HandTracker
from utils.gesture_utils import GestureDetector

class VirtualMouse:
    def __init__(self):
        """Initialize the Virtual Mouse application."""
        self.cap = cv2.VideoCapture(0)
        self.hand_tracker = HandTracker()
        self.gesture_detector = GestureDetector()
        self.prev_landmarks = []
        self.frame_reduction = 100  # Frame reduction for better performance
        self.smoothening = 7  # Smoothening factor for cursor movement
        
        # Initialize PyAutoGUI settings
        pyautogui.FAILSAFE = False  # Disable fail-safe
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize variables for cursor movement
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.prev_time = time.time()
        
        # Initialize emoji display variables
        self.current_emoji = None
        self.emoji_display_time = 0
        self.emoji_duration = 2.0  # Duration to display emoji in seconds
        self.emoji_size = 100  # Size of emoji in pixels
        
        # Map emojis to their image file paths
        self.emoji_image_paths = {
            "❤️": "assets/emojis/heart.png",
            "😊": "assets/emojis/smile.png",
            "👍": "assets/emojis/thumbs_up.png",
            "🤘": "assets/emojis/rock.png",
            "✌️": "assets/emojis/victory.png"
        }

        # Pre-load emoji images
        self.emoji_images = {
            emoji: self.load_emoji_image(path) 
            for emoji, path in self.emoji_image_paths.items()
        }

    def load_emoji_image(self, image_path):
        """Load and prepare an emoji image."""
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
        try:
            img = Image.open(full_path).convert("RGBA")
            img = img.resize((self.emoji_size, self.emoji_size), Image.LANCZOS)
            return np.array(img)
        except FileNotFoundError:
            print(f"Warning: Emoji image not found at {full_path}. Displaying placeholder.")
            # Create a placeholder image (e.g., a red square)
            placeholder = np.zeros((self.emoji_size, self.emoji_size, 4), dtype=np.uint8)
            placeholder[:, :, 2] = 255  # Red color
            placeholder[:, :, 3] = 128  # Semi-transparent
            return placeholder
        except Exception as e:
            print(f"Error loading emoji image {full_path}: {e}")
            placeholder = np.zeros((self.emoji_size, self.emoji_size, 4), dtype=np.uint8)
            placeholder[:, :, 0] = 255  # Blue color
            placeholder[:, :, 3] = 128
            return placeholder

    def handle_volume_control(self, volume_change):
        """Handle volume control gestures."""
        if volume_change == 1:
            pyautogui.press('volumeup')
        elif volume_change == -1:
            pyautogui.press('volumedown')

    def handle_screenshot(self):
        """Handle screenshot gesture."""
        pyautogui.hotkey('win', 'shift', 's')

    def handle_tab_switch(self, direction):
        """Handle tab switching gestures."""
        if direction == 1:
            pyautogui.hotkey('ctrl', 'tab')
        elif direction == -1:
            pyautogui.hotkey('ctrl', 'shift', 'tab')

    def handle_mic_toggle(self):
        """Handle mic toggle gesture."""
        pyautogui.press('f4')  # Assuming F4 is your mic mute key

    def display_emoji(self, frame, emoji):
        """Display emoji on the frame."""
        if emoji and time.time() - self.emoji_display_time < self.emoji_duration:
            emoji_img_rgba = self.emoji_images.get(emoji)
            if emoji_img_rgba is None:
                return
            
            # Calculate position to center the emoji on the frame
            frame_height, frame_width = frame.shape[:2]
            x_offset = (frame_width - self.emoji_size) // 2
            y_offset = (frame_height - self.emoji_size) // 2
            
            # Extract RGB and Alpha channels
            emoji_rgb = emoji_img_rgba[:, :, :3]
            alpha_channel = emoji_img_rgba[:, :, 3] / 255.0  # Normalize alpha to 0-1

            # Ensure the slice is within bounds and has correct dimensions
            h_slice = slice(y_offset, y_offset + self.emoji_size)
            w_slice = slice(x_offset, x_offset + self.emoji_size)
            
            # Get the region of interest from the frame
            roi = frame[h_slice, w_slice]

            # Apply alpha blending for transparency
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + emoji_rgb[:, :, c] * alpha_channel

            frame[h_slice, w_slice] = roi

    def run(self):
        """Main loop for the virtual mouse application."""
        print("Starting Virtual Mouse...")
        print("Press 'q' to quit")
        
        while True:
            # Read frame from webcam
            success, img = self.cap.read()
            if not success:
                print("Failed to grab frame")
                break
                
            # Flip the image horizontally for a later selfie-view display
            img = cv2.flip(img, 1)
            
            # Get the frame dimensions
            frame_height, frame_width, _ = img.shape
            
            # Find hands in the frame
            img, landmarks = self.hand_tracker.find_hands(img)
            
            if landmarks:
                # Get index finger tip position
                index_tip = landmarks[8]  # Index finger tip landmark
                
                # Map coordinates to screen
                x, y = self.gesture_detector.map_to_screen_coordinates(
                    index_tip[0], index_tip[1], frame_width, frame_height
                )
                
                # Smooth cursor movement
                x, y = self.gesture_detector.smooth_cursor_movement(x, y)
                
                # Move cursor
                pyautogui.moveTo(x, y)
                
                # Check for click gesture
                if self.gesture_detector.is_click_gesture(landmarks):
                    pyautogui.click()
                    time.sleep(0.2)  # Prevent multiple clicks
                
                # Check for right-click gesture
                if self.gesture_detector.is_right_click_gesture(landmarks):
                    pyautogui.rightClick()
                    time.sleep(0.2)  # Prevent multiple clicks
                
                # Check for scroll gesture
                if self.prev_landmarks:
                    scroll_direction = self.gesture_detector.is_scroll_gesture(
                        landmarks, self.prev_landmarks
                    )
                    if scroll_direction != 0:
                        pyautogui.scroll(scroll_direction * 10)
                
                # Check for volume control gesture
                volume_change = self.gesture_detector.is_volume_gesture(landmarks)
                if volume_change != 0:
                    self.handle_volume_control(volume_change)
                
                # Check for screenshot gesture
                if self.gesture_detector.is_screenshot_gesture(landmarks):
                    self.handle_screenshot()
                
                # Check for tab switch gesture
                if self.prev_landmarks:
                    tab_direction = self.gesture_detector.is_tab_switch_gesture(
                        landmarks, self.prev_landmarks
                    )
                    if tab_direction != 0:
                        self.handle_tab_switch(tab_direction)
                
                # Check for mic toggle gesture
                if self.gesture_detector.is_mic_toggle_gesture(landmarks):
                    self.handle_mic_toggle()
                
                # Check for emoji gestures
                current_time = time.time()
                if current_time - self.emoji_display_time > self.emoji_duration:
                    if self.gesture_detector.is_heart_emoji_gesture(landmarks):
                        self.current_emoji = "❤️"
                        self.emoji_display_time = current_time
                    elif self.gesture_detector.is_smile_emoji_gesture(landmarks):
                        self.current_emoji = "😊"
                        self.emoji_display_time = current_time
                    elif self.gesture_detector.is_thumbs_up_emoji_gesture(landmarks):
                        self.current_emoji = "👍"
                        self.emoji_display_time = current_time
                    elif self.gesture_detector.is_rock_emoji_gesture(landmarks):
                        self.current_emoji = "🤘"
                        self.emoji_display_time = current_time
                    elif self.gesture_detector.is_victory_emoji_gesture(landmarks):
                        self.current_emoji = "✌️"
                        self.emoji_display_time = current_time
                
                # Update previous landmarks
                self.prev_landmarks = landmarks
            
            # Display emoji if active
            self.display_emoji(img, self.current_emoji)
            
            # Display FPS
            cv2.putText(img, f"FPS: {int(1/(time.time() - self.prev_time))}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Virtual Mouse", img)
            
            # Update previous time
            self.prev_time = time.time()
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    virtual_mouse = VirtualMouse()
    virtual_mouse.run() 
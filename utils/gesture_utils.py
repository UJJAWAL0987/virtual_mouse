import numpy as np
import cv2
import pyautogui
from collections import deque
import keyboard
import time
import mediapipe as mp

class GestureDetector:
    def __init__(self, smoothing_factor=5):
        """
        Initialize the gesture detector.
        
        Args:
            smoothing_factor (int): Number of frames to use for smoothing cursor movement
        """
        self.smoothing_factor = smoothing_factor
        self.prev_points = deque(maxlen=smoothing_factor)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # MediaPipe hand landmark indices
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20
        self.WRIST = 0
        
        # Gesture cooldown timers
        self.last_volume_change = 0
        self.last_screenshot = 0
        self.last_tab_switch = 0
        self.last_mic_toggle = 0
        self.cooldown = 0.5  # Cooldown time in seconds
        self.prev_landmarks = None
        self.prev_time = time.time()
        self.last_gesture_time = 0
        self.emoji_cooldown = 1.0  # Longer cooldown for emoji gestures

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points."""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def is_click_gesture(self, landmarks):
        """
        Detect if the thumb and index finger are touching (click gesture).
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            bool: True if click gesture is detected
        """
        if len(landmarks) < 21:  # Need at least 21 landmarks for a complete hand
            return False
            
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 30  # Threshold for click detection

    def is_right_click_gesture(self, landmarks):
        """
        Detect if thumb, index, and middle fingers form a circle (right-click gesture).
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            bool: True if right-click gesture is detected
        """
        if len(landmarks) < 21:
            return False
            
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.MIDDLE_FINGER_TIP]
        
        # Check if all three fingers are close to each other
        dist1 = self.calculate_distance(thumb_tip, index_tip)
        dist2 = self.calculate_distance(thumb_tip, middle_tip)
        dist3 = self.calculate_distance(index_tip, middle_tip)
        
        return dist1 < 40 and dist2 < 40 and dist3 < 40

    def is_volume_gesture(self, landmarks):
        """
        Detect volume up/down gesture using thumb pointing up/down.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            int: 1 for volume up, -1 for volume down, 0 for no change
        """
        if len(landmarks) < 21:
            return 0
            
        current_time = time.time()
        if current_time - self.last_volume_change < self.cooldown:
            return 0
            
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[2]  # Thumb MCP joint
        
        # Check if thumb is pointing up or down
        if thumb_tip[1] < thumb_mcp[1]:  # Thumb pointing up
            self.last_volume_change = current_time
            return 1
        elif thumb_tip[1] > thumb_mcp[1]:  # Thumb pointing down
            self.last_volume_change = current_time
            return -1
        return 0

    def is_screenshot_gesture(self, landmarks):
        """
        Detect 'L' shape gesture for taking screenshot.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            bool: True if screenshot gesture is detected
        """
        if len(landmarks) < 21:
            return False
            
        current_time = time.time()
        if current_time - self.last_screenshot < self.cooldown:
            return False
            
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        wrist = landmarks[self.WRIST]
        
        # Calculate angle between thumb and index finger
        angle = self.calculate_angle(thumb_tip, wrist, index_tip)
        
        # Check if angle is approximately 90 degrees (L shape)
        if 75 <= angle <= 105:
            self.last_screenshot = current_time
            return True
        return False

    def is_tab_switch_gesture(self, landmarks, prev_landmarks):
        """
        Detect left/right swipe with two fingers for switching tabs.
        
        Args:
            landmarks: Current hand landmarks
            prev_landmarks: Previous frame's hand landmarks
            
        Returns:
            int: 1 for next tab, -1 for previous tab, 0 for no switch
        """
        if len(landmarks) < 21 or len(prev_landmarks) < 21:
            return 0
            
        current_time = time.time()
        if current_time - self.last_tab_switch < self.cooldown:
            return 0
            
        # Get index and middle finger tips
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.MIDDLE_FINGER_TIP]
        prev_index_tip = prev_landmarks[self.INDEX_FINGER_TIP]
        prev_middle_tip = prev_landmarks[self.MIDDLE_FINGER_TIP]
        
        # Calculate average horizontal movement
        current_x = (index_tip[0] + middle_tip[0]) / 2
        prev_x = (prev_index_tip[0] + prev_middle_tip[0]) / 2
        
        movement = current_x - prev_x
        
        if abs(movement) > 50:  # Threshold for tab switch
            self.last_tab_switch = current_time
            return 1 if movement > 0 else -1
        return 0

    def is_mic_toggle_gesture(self, landmarks):
        """
        Detect mic toggle gesture (pinch gesture with thumb and index finger).
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            bool: True if mic toggle gesture is detected
        """
        if len(landmarks) < 21:
            return False
            
        current_time = time.time()
        if current_time - self.last_mic_toggle < self.cooldown:
            return False
            
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        
        distance = self.calculate_distance(thumb_tip, index_tip)
        
        if distance < 20:  # Threshold for pinch gesture
            self.last_mic_toggle = current_time
            return True
        return False

    def is_scroll_gesture(self, landmarks, prev_landmarks):
        """
        Detect upward/downward swiping motion for scrolling using index and middle fingers.
        
        Args:
            landmarks: Current hand landmarks
            prev_landmarks: Previous frame's hand landmarks
            
        Returns:
            int: 1 for scroll up, -1 for scroll down, 0 for no scroll
        """
        if len(landmarks) < 21 or len(prev_landmarks) < 21:
            return 0
            
        # Get index and middle finger tips
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.MIDDLE_FINGER_TIP]
        prev_index_tip = prev_landmarks[self.INDEX_FINGER_TIP]
        prev_middle_tip = prev_landmarks[self.MIDDLE_FINGER_TIP]
        
        # Calculate average vertical movement of both fingers
        current_y = (index_tip[1] + middle_tip[1]) / 2
        prev_y = (prev_index_tip[1] + prev_middle_tip[1]) / 2
        
        # Calculate vertical movement
        movement = prev_y - current_y
        
        # Check if fingers are extended (not curled)
        index_mcp = landmarks[5]  # Index finger MCP joint
        middle_mcp = landmarks[9]  # Middle finger MCP joint
        
        # Calculate if fingers are extended
        index_extended = index_tip[1] < index_mcp[1]
        middle_extended = middle_tip[1] < middle_mcp[1]
        
        if abs(movement) > 30 and index_extended and middle_extended:  # Threshold for scroll detection
            return 1 if movement > 0 else -1
        return 0

    def smooth_cursor_movement(self, x, y):
        """
        Apply smoothing to cursor movement using moving average.
        
        Args:
            x: Current x coordinate
            y: Current y coordinate
            
        Returns:
            tuple: (smoothed_x, smoothed_y)
        """
        self.prev_points.append((x, y))
        
        if len(self.prev_points) < self.smoothing_factor:
            return x, y
            
        avg_x = sum(p[0] for p in self.prev_points) / len(self.prev_points)
        avg_y = sum(p[1] for p in self.prev_points) / len(self.prev_points)
        
        return int(avg_x), int(avg_y)

    def map_to_screen_coordinates(self, x, y, frame_width, frame_height):
        """
        Map camera coordinates to screen coordinates.
        
        Args:
            x: Camera x coordinate
            y: Camera y coordinate
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
            
        Returns:
            tuple: (screen_x, screen_y)
        """
        screen_x = int(np.interp(x, (0, frame_width), (0, self.screen_width)))
        screen_y = int(np.interp(y, (0, frame_height), (0, self.screen_height)))
        return screen_x, screen_y

    def is_heart_emoji_gesture(self, landmarks):
        """Detect heart emoji gesture (both hands forming a heart shape)."""
        if not landmarks:
            return False
            
        # Get thumb and index finger positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index
        distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        
        # Check if fingers are close enough to form a heart shape
        return distance < 40

    def is_smile_emoji_gesture(self, landmarks):
        """Detect smile emoji gesture (index and middle fingers curved up)."""
        if not landmarks:
            return False
            
        # Get finger positions
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        
        # Check if fingers are curved upward
        index_curved = index_tip[1] < index_mcp[1]
        middle_curved = middle_tip[1] < middle_mcp[1]
        
        return index_curved and middle_curved

    def is_thumbs_up_emoji_gesture(self, landmarks):
        """Detect thumbs up emoji gesture (thumb pointing up)."""
        if not landmarks:
            return False
            
        # Get thumb positions
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        # Check if thumb is pointing up
        return thumb_tip[1] < thumb_mcp[1] and abs(thumb_tip[0] - thumb_mcp[0]) < 20

    def is_rock_emoji_gesture(self, landmarks):
        """Detect rock emoji gesture (index and pinky fingers extended)."""
        if not landmarks:
            return False
            
        # Get finger positions
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        
        # Check if index and pinky are extended while others are curled
        index_extended = index_tip[1] < index_mcp[1]
        pinky_extended = pinky_tip[1] < pinky_mcp[1]
        
        return index_extended and pinky_extended

    def is_victory_emoji_gesture(self, landmarks):
        """Detect victory emoji gesture (index and middle fingers in V shape)."""
        if not landmarks:
            return False
            
        # Get finger positions
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        
        # Check if index and middle fingers are extended in V shape
        index_extended = index_tip[1] < index_mcp[1]
        middle_extended = middle_tip[1] < middle_mcp[1]
        
        # Check if fingers are spread apart
        finger_distance = np.sqrt((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)
        
        return index_extended and middle_extended and finger_distance > 50 
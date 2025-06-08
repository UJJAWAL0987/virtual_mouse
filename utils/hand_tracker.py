import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand tracker with MediaPipe Hands.
        
        Args:
            mode (bool): Whether to treat the input images as a batch of static images
            max_hands (int): Maximum number of hands to detect
            detection_confidence (float): Minimum confidence value for hand detection
            tracking_confidence (float): Minimum confidence value for hand tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        """
        Find hands in the image and return the image with hand landmarks drawn.
        
        Args:
            img: Input image
            draw (bool): Whether to draw the hand landmarks
            
        Returns:
            img: Image with hand landmarks drawn
            landmarks: List of hand landmarks
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        landmarks = []

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark positions
                for lm in hand_landmarks.landmark:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))

        return img, landmarks

    def get_landmark_position(self, img, landmark_id):
        """
        Get the position of a specific landmark.
        
        Args:
            img: Input image
            landmark_id: ID of the landmark to get position for
            
        Returns:
            tuple: (x, y) coordinates of the landmark
        """
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                h, w, c = img.shape
                lm = hand_landmarks.landmark[landmark_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                return cx, cy
        return None 
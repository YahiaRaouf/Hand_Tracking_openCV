import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # Mode for static or video input
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Minimum detection confidence
        self.trackCon = trackCon  # Minimum tracking confidence

        self.mphands = mp.solutions.hands  # Initialize MediaPipe hands solution
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpdraw = mp.solutions.drawing_utils  # Initialize drawing utilities

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        results = self.hands.process(imgRGB)  # Process the image and detect hands

        if results.multi_hand_landmarks:  # If hands are detected
            for handLms in results.multi_hand_landmarks:  # Iterate through detected hands
                if draw:  # If drawing is enabled
                    self.mpdraw.draw_landmarks(
                        img, handLms, self.mphands.HAND_CONNECTIONS  # Draw landmarks and connections
                    )
        return img  # Return the image with or without drawings

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  # List to store landmark positions
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Process the image and detect hands
        if results.multi_hand_landmarks:  # If hands are detected
            myHand = results.multi_hand_landmarks[handNo]  # Get the specified hand
            for id, lm in enumerate(myHand.landmark):  # Iterate through landmarks
                height, width, channel = img.shape  # Get image dimensions
                cx, cy = int(lm.x * width), int(lm.y * height)  # Calculate pixel coordinates
                lmList.append([id, cx, cy])  # Append landmark ID and coordinates to the list
                if draw:  # If drawing is enabled
                    if id == 4 or id == 8:  # Draw circles on landmarks 4 and 8
                        cv2.circle(img, (cx, cy), 8, (2, 7, 93), cv2.FILLED)
        return lmList  # Return the list of landmark positions

import cv2
import time
import HandTrackingModule as htm  # Import the hand tracking module


def main():
    prevTime = 0  # Initialize previous time for FPS calculation
    currentTime = 0  # Initialize current time for FPS calculation

    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open video capture.")  # Print error if camera cannot be opened
        return

    detector = htm.handDetector(detectionCon=0.5)  # Create an instance of handDetector with detection confidence of 0.5

    while True:
        success, img = cap.read()  # Capture frame-by-frame
        if not success:
            print("Error: Failed to capture image.")  # Print error if frame cannot be captured
            break

        img = detector.findHands(img)  # Detect hands in the frame
        lmList = detector.findPosition(img)  # Get the list of landmark positions
        if len(lmList) != 0:
            print(lmList[4], lmList[8])  # Print the positions of landmarks 4 and 8 if landmarks are detected

        currentTime = time.time()  # Get the current time
        fps = 1 / (currentTime - prevTime)  # Calculate frames per second
        prevTime = currentTime  # Update previous time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 255, 0), 3)  # Display FPS on the frame

        cv2.imshow("Image", img)  # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop if 'q' is pressed
            break

    cap.release()  # Release the capture
    cv2.destroyAllWindows()  # Destroy all OpenCV windows


if __name__ == "__main__":
    main()  # Run the main function if this script is executed

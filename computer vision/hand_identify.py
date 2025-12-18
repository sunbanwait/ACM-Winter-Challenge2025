import cv2
import numpy as np
import platform

# --- 1. GLOBAL VARIABLES & CONSTANTS ---
FRAME_HEIGHT = 500
FRAME_WIDTH = 900
CALIBRATION_TIME = 30 
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

# Define the region of interest (The Blue Box)
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH

# Global background variable
background = None

# --- 2. THE HAND DATA CLASS ---
class HandData:
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        self.isInFrame = False
        self.isWaving = False
        self.fingers = 0

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

# --- 3. PROCESSING FUNCTIONS (New additions) ---

def get_region(frame):
    # Crop the frame to the blue box area
    region = frame[region_top:region_bottom, region_left:region_right]
    # Convert to grayscale and blur to remove noise (better for edge detection)
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region, (5,5), 0)
    return region

def get_average(region):
    global background
    # Initialize background if first frame
    if background is None:
        background = region.copy().astype("float")
        return
    # Accumulate weighted average of background
    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(region):
    global hand
    # Calculate difference between background and current frame
    diff = cv2.absdiff(background.astype(np.uint8), region)
    
    # Threshold to get binary image (white = hand, black = background)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    # Get contours (outlines) of the hand
    # Note: Using robust syntax for different OpenCV versions
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        if hand is not None: hand.isInFrame = False
        return
    else:
        if hand is not None: hand.isInFrame = True
        # Return the largest contour (the hand)
        segmented_region = max(contours, key=cv2.contourArea)
        return (thresholded_region, segmented_region)

def write_on_image(frame, hand, frames_elapsed):
    text = "Searching..."

    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand is None or hand.isInFrame == False:
        text = "No hand detected"
    else:
        if hand.isWaving: text = "Waving"
        elif hand.fingers == 0: text = "Rock"
        elif hand.fingers == 1: text = "Pointing"
        elif hand.fingers == 2: text = "Scissors"
    
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    frames_elapsed = 0
    # FIX: Initialize the hand object with dummy data so it exists
    hand = HandData(top=(0,0), bottom=(0,0), left=(0,0), right=(0,0), centerX=0)

    print("Starting camera...")
    
    # Platform-specific camera setup
    system = platform.system()
    camera_index = 0 
    if system == "Darwin":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    elif system == "Windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"ERROR: Camera could not be opened.")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize and Flip
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)

            # --- NEW LOGIC START ---
            # 1. Get the processed region (grayscale/blurred)
            region = get_region(frame)

            # 2. Calibration Phase vs Detection Phase
            if frames_elapsed < CALIBRATION_TIME:
                get_average(region)
            else:
                # Attempt to find the hand
                region_pair = segment(region)
                if region_pair is not None:
                    (thresholded_region, segmented_region) = region_pair
                    # Draw the hand outline on the main frame (adjusting coordinates to match global frame)
                    # We add (region_left, region_top) because the contour is relative to the small box
                    cv2.drawContours(frame, [segmented_region + (region_left, region_top)], -1, (255, 255, 255))
                    
                    # Show the binary mask in a separate window (for debugging)
                    cv2.imshow("Segmented Image", thresholded_region)
            # --- NEW LOGIC END ---

            write_on_image(frame, hand, frames_elapsed)
            cv2.imshow("Hand Gesture Recognition", frame)

            frames_elapsed += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")
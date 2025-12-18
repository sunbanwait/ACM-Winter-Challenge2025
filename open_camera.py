import cv2
import platform

system=platform.system()

if system == "Darwin":  # macOS
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
elif system == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:  # Linux
    cap = cv2.VideoCapture(0)
    
# Use macOS native camera backend
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Un-mirror macOS camera feed
    frame = cv2.flip(frame, 1)

    cv2.imshow("GestureCamino - macOS Camera", frame)

    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")

import cv2
import platform

system=platform.system()

if system == "Darwin":  # mac
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
elif system == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:  # Linux
    cap = cv2.VideoCapture(0)

print("Starting camera... ")
print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    if system == "Darwin":
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    elif system == "Windows":
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap=cv2.VideoCapture(1)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Camera failed.")
        cap.release()
        exit()
        
print("Camera is working!")
while True:
    # Un-mirror camera feed
    frame = cv2.flip(frame, 1)
    cv2.imshow("Camera", frame)

    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break
    ret, frame = cap.read()
    if not ret:
        print("Lost camera feed")
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")

import cv2
import sys

# Load the Haar Cascade for face detection
cascPath = "C:/Users/User/Downloads/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Image path should be provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py imagePath")
    sys.exit(1)

imagePath = sys.argv[1]
image = cv2.imread(imagePath)
if image is None:
    print("Error: Image could not be read")
    sys.exit(1)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(25, 25)
)

# Draw rectangles around each face
for (x, y, width, height) in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Display the result
cv2.imshow('Faces found', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

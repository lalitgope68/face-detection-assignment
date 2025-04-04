import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = r"C:\Users\lalit\Documents\face detection\face dtect app\family.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Image not loaded!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#  Detect faces (tuned)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.07,
    minNeighbors=5,
    minSize=(60, 60)
)

# Applied NMS to remove duplicate boxes
def apply_nms(boxes, threshold=0.3):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(y2)

    keep = []

    while len(order) > 0:
        i = order[-1]
        keep.append(i)
        order = order[:-1]

        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / areas[order]
        order = order[overlap <= threshold]

    return boxes[keep]

faces_nms = apply_nms(faces)

 # Draw rectangles
for (x, y, w, h) in faces_nms:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show output
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("output")
plt.axis("off")
plt.show()

# Print face count
print(" Faces detected (cleaned):", len(faces_nms))


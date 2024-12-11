import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load DNN face detection model (Caffe)
modelFile = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = cv2.data.haarcascades + "deploy.prototxt"

# Load the model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Read the input image
image = cv2.imread('1.jpg')
(h, w) = image.shape[:2]

# Prepare the image for the neural network
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), 
                             mean=(104.0, 177.0, 123.0))

# Perform face detection
net.setInput(blob)
detections = net.forward()

# Draw detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    # Filter out weak detections
    if confidence > 0.5:  # Adjust this threshold if needed
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Draw bounding box
        text = f"{confidence*100:.2f}%"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Face Detection using DNN")
plt.axis("off")
plt.show()

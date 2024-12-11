import cv2
import numpy as np
import matplotlib.pyplot as plt


modelFile = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = cv2.data.haarcascades + "deploy.prototxt"


net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


image = cv2.imread('1.jpg')
(h, w) = image.shape[:2]


blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), 
                             mean=(104.0, 177.0, 123.0))


net.setInput(blob)
detections = net.forward()


for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    
    if confidence > 0.5:  
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        
        text = f"{confidence*100:.2f}%"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Face Detection using DNN")
plt.axis("off")
plt.show()

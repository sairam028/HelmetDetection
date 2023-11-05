    #import cv2
    #import numpy as np
    #import random
    #import os
    from PIL import Image
    #import time
    #import imutils
    from tensorflow.keras.models import load_model
    #import pytesseract  
    # Import Tesseract OCR library
    
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    model = load_model('helmet_nonhelmet_cnn.h5')  # Fixed model filename typo
    print('Model loaded!!!')
    
    cap = cv2.VideoCapture('testing_videos/test2.mp4')
    COLORS = [(0, 255, 0), (0, 0, 255)]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Fixed codec specifier
    
    writer = cv2.VideoWriter('output.avi', fourcc, 5, (888, 588))
    writer.open()

# Function to perform OCR on an image
def perform_ocr(image):
    try:
        # Convert the image to grayscale for OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
   # Perform OCR using pytesseract
        text = pytesseract.image_to_string(gray)
        
        return text
    except:
        return ""

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

ret = True

while ret:
    ret, img = cap.read()
    img = imutils.resize(img, height=500)  # Fixed height assignment
    
   height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

 # Perform OCR on the detected regions
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        roi = img[y:y + h, x:x + w]
        text = perform_ocr(roi)
        
   # Draw bounding box and label text on the image
        color = COLORS[classIds[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"Class {classIds[i]} - {text}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Write the processed frame to the output video
    writer.write(img)

# Release video writer and capture objects
writer.release()
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

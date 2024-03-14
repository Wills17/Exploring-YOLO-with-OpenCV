from ultralytics import YOLO
import cv2 as cv

print("0\33c") #Clear screen

# Define YOLO model
model = YOLO("yolov8s-world.pt")

# Read image
image = cv.imread("People.jpg")
print("Image of people has been produced")
cv.imshow("Image showing people", image)

# Store image in a variable - result
result = model(image)
print()

# Collect and display results
for i, r in enumerate(result):
    detection = r.boxes.data.tolist()
    names = r.names
    classes = r.boxes.cls.tolist()
    
    # List to store detected people in images.
    list = []
    
    for labels, detections in zip(classes, detection):
        
        # To ensure only people in the image are detected
        if labels == 0:
            label = names[labels]
            list.append(label)
            
            x,y,w,h,conf, _ = detections
            # Round confidence value to 3 decimal places
            conf = round(conf, 3)
            
            cv.putText(image, str(label), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv.LINE_AA)
            cv.putText(image, str(conf), (int(x), int(h)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)
            cv.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0,100,0), 2)
            
        
print("Total number of people detected in image:", len(list))
cv.imshow("Image showing detecting objects", image)
cv.waitKey(0)
cv.destroyAllWindows()

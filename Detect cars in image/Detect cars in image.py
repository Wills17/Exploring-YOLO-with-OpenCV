from ultralytics import YOLO
import cv2 as cv

print("0\33c") #Clear screen

# Define YOLO model
model = YOLO("yolov8n.pt")

# Read image
image = "Cars.jpg"
image = cv.imread(image)
print("Image of cars has been displayed")
cv.imshow("Image of cars", image)


# Store image in a variable - result
result = model(image, save=True)
print()

# Collect and display results
for i, r in enumerate(result):
    detection = r.boxes.data.tolist()
    names = r.names
    classes = r.boxes.cls.tolist()
    
    # List to store detected people in images.    
    list = []
    
    for labels, detections in zip(classes, detection):
            label = names[labels]
            list.append(label)
            
            x,y,w,h,conf, _ = detections
            # Round confidence value to 3 decimal places
            conf = round(conf, 3)
            
            cv.putText(image, str(label), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv.LINE_AA)
            cv.putText(image, str(conf), (int(x), int(h)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
            cv.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0,100,0), 2)
            
        
print("Total number of cars detected in image:", len(list))
cv.imshow("Image showing detecting objects", image)
cv.waitKey(0)
cv.destroyAllWindows()

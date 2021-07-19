import webbrowser
import cv2
import numpy as np
import html
import html_img


net = cv2.dnn.readNet("gokul.weights", "gokul.cfg")
classes =[]
with open("gokul.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))

img = cv2.imread("gokul.jpg")
img = cv2.resize(img,None, fx=1, fy=1)
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392,(416,416),(0,0,0),True, crop= False)


net.setInput(blob)
outs = net.forward(output_layers)
class_ids = []
confidences =[]
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            cv2.rectangle(img, (x, y),(x + w, y + h), (255,0,0),2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id )
indexes = cv2.dnn.NMSBoxes(boxes,confidences, 0.5, 0.4)
print(indexes)
font= cv2.FONT_HERSHEY_PLAIN
with open('new.html', 'w') as outfile:
 for i in range(len(boxes)):
    if i in indexes:
      x,y,w,h = boxes[i]
      label = classes[class_ids[i]]
      color = colors[i]
      cv2.rectangle(img,(x,y),(x + w, y+h), (0, 255,0),2)
      cv2.putText(img, label, (x,y + 30), font, 3,(0,0,0),3)
      html= f"<html> <head> </head> <body><h2> DETECTING IMAGES : {label} </h2>  </body> </html>"
      outfile.write(html)
cv2.imshow("image", img)
cv2.imwrite("image1.jpg",img)
webbrowser.open_new_tab("new.html")
cv2.waitKey(0)
cv2.destroyAllWindows()


from sort_algo import *
import random
import cv2
from ultralytics import YOLO

def load_classes(filename):
    classes = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        classes.append(line)
    return classes

def detect(ptfile):
    net = YOLO(ptfile)
    mot_tracker = Sort()
    filename = 'coco.names'
    classes = load_classes(filename)
    cap = cv2.VideoCapture('people_walk.mp4')
    color_list = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(1000)]
    
    # Start
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        results = net(img)
        boxes = []
        confidences = []
        classIDs = []
        for result in results:
            for obj in result.boxes:
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                confidence = obj.conf[0]
                classID = int(obj.cls[0])
                if confidence > 0.5:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        result_img = np.copy(img)
        dets = []
        count_detection = 0
        for i in idxs:
            # i = i[0]
            if classes[classIDs[i]] == 'person':
                count_detection += 1
        
        if count_detection > 0:
            detects = np.zeros((count_detection, 5))
            count = 0
            for i in idxs:
                # i = i[0]
                b = boxes[i]
                if classes[classIDs[i]] == 'person':
                    x1, y1, w, h = b
                    x2, y2 = x1 + w, y1 + h
                    detects[count, :] = [x1, y1, x2, y2, confidences[i]]
                    count += 1
            
            if len(detects) != 0:
                trackers = mot_tracker.update(detects)
                for d in trackers:
                    result_img = cv2.rectangle(result_img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color_list[int(d[4]) % 1000], 2)
        
        cv2.imshow('demo', result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect('yolov8n.pt')

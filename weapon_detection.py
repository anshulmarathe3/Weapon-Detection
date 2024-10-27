import cv2
import numpy as np

# Load YOLO network with trained weights and config
yolo_net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
class_labels = ["Weapon"]

# Uncomment if using a file with class names
# with open("coco.names", "r") as f:
#     class_labels = [line.strip() for line in f.readlines()]

# Get the output layer names
output_layer_names = yolo_net.getUnconnectedOutLayersNames()
colors_palette = np.random.uniform(0, 255, size=(len(class_labels), 3))

# Prompt for video file name or default to webcam
def get_video_source():
    source = input("Enter file name or press enter to start webcam: \n")
    return 0 if source == "" else source

# Capture video from source
video_capture = cv2.VideoCapture(get_video_source())

# Video processing loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to read a frame from the video source.")
        break
    frame_height, frame_width, channels = frame.shape

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    layer_outputs = yolo_net.forward(output_layer_names)

    # Process detection results
    detected_class_ids = []
    detection_confidences = []
    bounding_boxes = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Calculate object bounding box coordinates
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                box_width = int(detection[2] * frame_width)
                box_height = int(detection[3] * frame_height)
                box_x = int(center_x - box_width / 2)
                box_y = int(center_y - box_height / 2)

                bounding_boxes.append([box_x, box_y, box_width, box_height])
                detection_confidences.append(float(confidence))
                detected_class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant boxes
    selected_boxes = cv2.dnn.NMSBoxes(bounding_boxes, detection_confidences, 0.5, 0.4)
    print(selected_boxes)
    if selected_boxes == 0:
        print("Weapon detected in frame")
    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(bounding_boxes)):
        if i in selected_boxes:
            x, y, w, h = bounding_boxes[i]
            label = str(class_labels[detected_class_ids[i]])
            color = colors_palette[detected_class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

    # Display the video frame with detections
    cv2.imshow("Detection Frame", frame)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

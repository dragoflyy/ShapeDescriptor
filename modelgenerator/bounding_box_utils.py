import numpy as np
import cv2

def detect_and_draw_bounding_boxes(image, prediction, threshold=0.5):
    # Convert the prediction to binary using the threshold
    binary_prediction = (prediction > threshold).astype(np.uint8) * 255

    # Find contours of the detected shapes in the binary image
    contours, _ = cv2.findContours(binary_prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure the input image is of type uint8 (required by OpenCV)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a copy of the input image to draw bounding boxes
    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    dimensions = []

    # Loop through each detected contour
    for contour in contours:
        # Calculate the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        dimensions.append((x, y, w, h))  # Append the bounding box dimensions to the list

        # Draw the bounding box on the image
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Return the image with bounding boxes and the dimensions of the detected objects
    return image_with_boxes, dimensions
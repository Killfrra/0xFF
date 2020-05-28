# Import required modules
import cv2 as cv
import math
import argparse
import numpy

############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

# Load network
model = 'frozen_east_text_detection.pb'
net = cv.dnn.readNet(model)

outNames = []
outNames.append("feature_fusion/Conv_7/Sigmoid")
outNames.append("feature_fusion/concat_3")


def get_rotated_boxes(image, confThreshold=0.9, nmsThreshold=0.1):

    inpWidth  = 640 #image.size[0] // 32 * 32
    inpHeight = 320 #image.size[1] // 32 * 32
    frame = cv.cvtColor(numpy.array(image), cv.COLOR_RGB2BGR)
    
    # Get frame height and width
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    rW = width_  / inpWidth
    rH = height_ / inpHeight

    # Create a 4D blob from frame.
    blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    # Run the model
    net.setInput(blob)
    outs = net.forward(outNames)

    # Get scores and geometry
    scores = outs[0]
    geometry = outs[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)

    outputs = []

    # Apply NMS
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH

        outputs.append((
            (vertices[0][0], vertices[0][1]),
            (vertices[1][0], vertices[1][1]),
            (vertices[2][0], vertices[2][1]),
            (vertices[3][0], vertices[3][1]),
        ))

    return outputs

def get_crop_boxes(image):
    outputs = []
    boxes = get_rotated_boxes(image)
    for (p1, p2, p3, p4) in boxes:
        left  = round(min(p1[0], p2[0], p3[0], p4[0]))
        upper = round(min(p1[1], p2[1], p3[1], p4[1]))
        right = round(max(p1[0], p2[0], p3[0], p4[0]))
        lower = round(max(p1[1], p2[1], p3[1], p4[1]))
        outputs.append((left, upper, right, lower))
    return outputs


if __name__ == "__main__":
    from PIL import Image
    import sys
    outputs = get_crop_boxes(Image.open(sys.argv[1]))
    print(outputs)
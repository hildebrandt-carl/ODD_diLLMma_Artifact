import cv2
import math
import numpy as np


def calculate_endpoint(starting_point, length, rotation):
    radians = math.radians(90-rotation)
    x = int(starting_point[0] + length * math.cos(radians))
    y = int(starting_point[1] - length * math.sin(radians))
    return [x, y]


def rotate_point(center, point, angle):
    """
    Rotate a point around a circle centered at 'center' by 'angle' degrees.

    Args:
        center: An tuple (x,y) which represents the center of the circle
        point: An tuple (x,y) which represents the point
        angle: The angle in degrees describing how much to rotate the point (negative is right, positive left)

    Returns:
        new_x: The rotated point x value
        new_y: The rotated point y value
    """
    cx, cy = center
    px, py = point

    # Convert angle from degrees to radians
    angle_rad = math.radians(-angle)

    # Calculate the new position of the point
    new_x = cx + math.cos(angle_rad) * (px - cx) - math.sin(angle_rad) * (py - cy)
    new_y = cy + math.sin(angle_rad) * (px - cx) + math.cos(angle_rad) * (py - cy)

    # Make sure they are rounded to integers
    new_x = int(round(new_x, 0))
    new_y = int(round(new_y, 0))

    return new_x, new_y

def show_steering(img, steering_angles, colors, labels=None, CLIPPING_DEGREE=90, frame_id=None):

    # Make sure everything adds up
    assert(np.shape(steering_angles)[0] == len(colors))

    steering_angles_clipped = np.clip(steering_angles, -CLIPPING_DEGREE+2, CLIPPING_DEGREE-2)

    # Add the label
    if labels is not None:
        assert(len(labels) == len(colors))
        for i in range(len(labels)):
            # Add text to the frame
            if steering_angles_clipped[i] >= 0:
                text = "{}: +{:.2f} deg".format(labels[i], np.round(steering_angles_clipped[i],2))
            else:
                text = "{}: {:.2f} deg".format(labels[i], np.round(steering_angles_clipped[i],2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.0
            color = colors[i]
            thickness = 7
            org = (75, 75+(100*i))

            # Place the steering angle
            img = cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    middle_x = int(round(np.shape(img)[1]/2, 0))
    middle_y = int(round(np.shape(img)[0]/2, 0))
    bottom_y = int(round(np.shape(img)[0], 0))

    for i in range(len(labels)):
        # Define the starting and ending points for the arrow
        start_point = (middle_x, bottom_y)
        end_point = (middle_x, middle_y)

        # Rotate the end point by steering angle
        end_point = rotate_point(start_point, end_point, steering_angles_clipped[i])

        # Set the color and thickness of the arrow
        color = colors[i]
        thickness = 20

        # Draw the arrow on the frame
        img = cv2.arrowedLine(img, start_point, end_point, color, thickness)

    if frame_id is not None:
        pos = (1550, 40)
        img = cv2.putText(img, "Frame ID: {:08d}".format(frame_id), pos, font, font_scale//1.75, (0,0,0), thickness//4, cv2.LINE_AA)

    return img
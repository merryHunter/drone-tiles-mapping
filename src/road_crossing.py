import cv2
import numpy as np
import math
from collections import defaultdict
import logging
logging.getLogger().setLevel(logging.DEBUG)


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def get_lines(src):
    dst = cv2.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    return cdst, lines


def try_get_road_crossing_center(mask: np.ndarray):
    # mask = cv2.resize(mask, (328, 220))  #todo: check if it works better with resized image
    vis, lines = get_lines(mask)
    try:
        segmented = segment_by_angle_kmeans(lines)
        intersections = segmented_intersections(segmented)
        center = np.mean(intersections, axis=0).astype(int)
        if center[0][0] < 0 or center[0][1] < 0:
            return False, lines
        return True, center[0]      
    except Exception as e:
        print(e)
    return False, lines

def get_road_crossings_img(mask):
    mask = cv2.resize(mask, (128, 100))
    vis, lines = get_lines(mask)
    try:
        segmented = segment_by_angle_kmeans(lines)
        intersections = segmented_intersections(segmented)
        center = np.mean(intersections, axis=0).astype(int)
        vis = cv2.circle(vis, (center[0],center[1]), radius=15, color=(0, 255, 0), thickness=4)
        logging.info('crossing detected')
    except Exception as e:
        logging.info('crossing not detected')
        # logging.error(e)
    return vis

def calculate_distance(point_a: np.array, point_b: np.array):
    return np.linalg.norm(point_a - point_b)


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections
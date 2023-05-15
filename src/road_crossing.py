import cv2
import numpy as np
import math
from collections import defaultdict
import logging
logging.getLogger().setLevel(logging.INFO)


class RoadCrossingDetector:
    def __init__(self):
        pass
    
    def get_road_crossing_center(self, road_mask_gray: np.array) -> np.ndarray:
        # get by contours, then by hough lines, then combine results
        center_c = self._get_road_crossing_center_by_contour(road_mask_gray)
        center_l = self._get_road_crossing_center_by_lines(road_mask_gray)
        road_mask = cv2.cvtColor(road_mask_gray, cv2.COLOR_GRAY2BGR)
        if center_c is not None and center_l is not None:
            # distance = calculate_distance(center_c, center_l)
            # logging.info(f"distance between detected centers: {distance}")
            # mean_center = np.mean([center_c, center_l], axis=0).astype(int)
            return center_l
        return None

    def _get_road_crossing_center_by_contour(self, road_mask_gray: np.array)-> np.ndarray:
        bbox, center = None, None
        road_mask_gray = cv2.GaussianBlur(road_mask_gray, (5, 5), 0)
        img = cv2.cvtColor(road_mask_gray, cv2.COLOR_GRAY2BGR)
        _, thresh = cv2.threshold(road_mask_gray, 60, 255, cv2.THRESH_BINARY)
        contours , hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
            # if area of counter is big enough, then it is a crossing
            # logging.info(f"contour area: {cv2.contourArea(contour)}; {len(approx)}")
            if len(approx) > 8 and len(approx) < 15 and cv2.contourArea(contour) > 40000:
                cv2.drawContours(img, [approx], 0, (0, 100, 0), 5)
                M = cv2.moments(contour)
                if M['m00'] != 0.0:
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])
                    center = np.array([x, y])
        return center
    
    def _get_road_crossing_center_by_lines(self, road_mask_gray: np.ndarray) -> np.ndarray:
        # mask = cv2.resize(mask, (328, 220))  #todo: check if it works better with resized image
        center = None
        vis, lines = self._get_lines(road_mask_gray)
        try:
            segmented = self._segment_by_angle_kmeans(lines)
            intersections = self._segmented_intersections(segmented)
            center = np.mean(intersections, axis=0).astype(int)[0]
            
            if center[0] < 0 or center[1] < 0:
                return None
            cv2.circle(vis, tuple(center), 10, (0, 255, 0), -1)
            # cv2.imshow("vis", vis)
        except Exception as e:
            logging.debug("crossing not detected")
            # print(e)
            center = None
        return center
    
    def _get_lines(self, road_mask_gray):
        _, thresh = cv2.threshold(road_mask_gray, 60, 255, cv2.THRESH_BINARY)
        dst = cv2.Canny(thresh, 200, 300)
        kernel = np.ones((7, 7), np.uint8)
        dst = cv2.dilate(dst, kernel, iterations=1)
        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

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
    
    def _segment_by_angle_kmeans(self, lines, k=2, **kwargs):
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
    
    def _intersection(self, line1, line2):
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


    def _segmented_intersections(self, lines):
        """Finds the intersections between groups of lines."""

        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(self._intersection(line1, line2)) 

        return intersections


def calculate_distance(point_a: np.array, point_b: np.array):
    return np.linalg.norm(point_a - point_b)
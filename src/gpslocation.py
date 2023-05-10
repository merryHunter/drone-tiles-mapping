import numpy as np
import cv2
import imutils


class GPSLocationTracker:
    def __init__(self, init_lat, init_lon, init_alt):
        self.lat = init_lat
        self.lon = init_lon
        self.alt = init_alt
        # init kalman filter  
     
    def get_current_position_on_tile(self, drone_img_gray: np.ndarray, satellite_img_gray: np.ndarray) -> tuple:
        """
        Matches drone image with satellite tile by using scale-invariant template matching over Canny edges.
        Returns the current position (x,y) of the drone on the satellite reference tile.
        """
        img = cv2.resize(drone_img_gray, (640, 480))
        img = cv2.Canny(img, 100, 300)
        (tH, tW) = img.shape[:2]
        # to keep track of the matched region
        found = None
        for scale in np.linspace(0.1, 1.0, 10)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(satellite_img_gray, width = int(satellite_img_gray.shape[1] * scale))
            r = satellite_img_gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, img, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        return ((startX + endX) // 2,(startY + endY) //2)

    
    def get_current_gps_location(self, drone_img: np.ndarray):
        # TODO: implement
        pass
        
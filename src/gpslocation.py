import numpy as np
import cv2
import imutils

from kalmanfilter import KalmanFilter2D
import logging
logging.getLogger().setLevel(logging.DEBUG)


class GPSLocationTracker:
    """
    Track drone location (with kalman filter) by mapping drone image on satellite tile
    """
    def __init__(self, init_x_position=400, init_y_position=1150, faulty_distance_threshold=200, steps_init=5):
        self._init_kalman_filter()
        self.last_position = (init_x_position, init_y_position)
        self.current_position = (init_x_position, init_y_position)
        self.steps_init = steps_init
        self.faulty_distance_threshold = faulty_distance_threshold
        self.step = 0

    def _init_kalman_filter(self):
        # Define the initial state estimate of the object (unknown to the Kalman filter)
        initial_state_estimate = [0, 0]

        # Define the initial covariance estimate of the object (unknown to the Kalman filter)
        initial_covariance_estimate = [1, 0, 0, 1]

        # Define the process noise covariance matrix
        process_noise_covariance = [0.1, 0, 0, 0.1]

        # Define the measurement noise covariance matrix
        measurement_noise_covariance = [1, 0, 0, 1]

        # Create a KalmanFilter2D object with the defined parameters
        self.kf = KalmanFilter2D(initial_state_estimate, initial_covariance_estimate, process_noise_covariance, measurement_noise_covariance)

    def track_position(self, drone_img_gray: np.ndarray, satellite_img_gray: np.ndarray) -> tuple:
        """
        track with kalman filter the position of the drone on the satellite tile
        """
        self.last_position = self.current_position
        # get current position of drone on satellite tile
        predicted_position = self.get_current_position_on_tile(drone_img_gray, satellite_img_gray)
        # filter faulty positions if current position is too far away from last position
        distance = np.linalg.norm(np.array(predicted_position) - np.array(self.last_position))
        logging.debug(f"distance: {distance}")
        if distance < self.faulty_distance_threshold or self.step < self.steps_init:
            logging.debug("updated position")
            # predict next position using kalman filter
            self.kf.predict()
            # update kalman filter with current position
            self.kf.update(predicted_position)
            # return predicted position
            self.current_position = [int(self.kf.x[0]), int(self.kf.x[1])]
        self.step += 1
        
        return self.current_position, predicted_position, distance

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
        

class SatelliteTileDownloader:
    """
    
    """
    def __init__(self, mock_satellite_path: str = None):
        tiles = []
        for j in range(1, 36):
            tiles.append(cv2.imread(str(mock_satellite_path / f"{j}.png")))
        self.tiles = tiles

    def get_tile(self, lat: float, lon: float, zoom: int = 17, index=0):
        if index >= len(self.tiles) or index < 0:
            raise ValueError("index out of range")
        return self.tiles[index]

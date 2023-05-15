from enum import Enum
import cv2
import numpy as np
from road_crossing import RoadCrossingDetector
import logging
logging.getLogger().setLevel(logging.DEBUG)


class NavigationState(Enum):
    LOCATING_WAYPOINT = 1
    HEAD_TO_WAYPOINT = 2
    CROSSING_WAYPOINT = 3


class RoadCrossingNavigator:

    def __init__(self, input_waypoints_file: str, found_threshold=5):
        self.waypoints_list = self.get_waypoints_list(input_waypoints_file)
        self.next_waypoint_idx = 0
        self.nav_state = NavigationState.HEAD_TO_WAYPOINT
        self.nav_state_history = [] # push state every 5Hz
        self.tracker = cv2.TrackerCSRT_create()
        self.found_counter = 0
        self.found_counter_threshold = found_threshold
        self.tracking_started = False
        self.down_counter = 0
        self.detector = RoadCrossingDetector()
        self.centers_history = []

    def track_road_crossing(self, road_mask_gray):
        """
        Returns road crossing center coordinates if located, None otherwise.
        """
        bbox, road_crossing_center = None, None
        road_crossing_center = self.detector.get_road_crossing_center(road_mask_gray)
        # if found
        if road_crossing_center is not None:
            self.found_counter += 1
            self.down_counter = 0
        # if not found but was found before
        elif self.found_counter > 0:
            self.found_counter -= 1
            self.down_counter += 1
        # if not found and was not found before
        if road_crossing_center is None and self.down_counter > 3: # stop tracking
            self.found_counter = 0
            self.tracking_started = False
            del self.tracker
            self.tracker = cv2.TrackerCSRT_create()
        logging.debug(f"road crossing found_counter: {self.found_counter}")
        # if found for a while - start/continue tracking
        if self.found_counter > self.found_counter_threshold:
            if self.tracking_started:
                ret, bbox = self.tracker.update(road_mask_gray)
            else:
                logging.debug(f"starting tracking, {road_crossing_center}")
                bbox = [int(road_crossing_center[0] - 50), int(road_crossing_center[1] - 50), 100, 100]
                ret = self.tracker.init(road_mask_gray, bbox)
                self.tracking_started = True
                return bbox, road_crossing_center
        return bbox, road_crossing_center

    def get_waypoints_list(self, input_waypoints_file: str):
        """
        Returns list of waypoints from input file.
        """
        with input_waypoints_file.open(mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            waypoints_list = [(float(line.split(',')[0]), float(line.split(',')[1])) for line in lines]
        logging.info(f"waypoints list: {waypoints_list}")
        return waypoints_list
    
    # def get_navigation_state(self, drone_img_gray):
    #     """
    #     todo? 
    #     Returns the current navigation state of the drone.
        
    #     Workflow:
    #     - try locate road crossing
    #     - if located - start tracking its position with KalmanFilter. if no - nothing changes, keep looking
    #     - if current position is in close proximity to the next waypoint - switch to CROSSING_WAYPOINT state
    #     - if current state is crossing waypoint for 2 seconds - switch to HEAD_TO_NEXT_WAYPOINT state, and update next waypoint index, reset kalman filter
    #     """
    #     pass
    
    def get_next_waypoint(self):
        return self.waypoints_list[self.next_waypoint_idx]
    
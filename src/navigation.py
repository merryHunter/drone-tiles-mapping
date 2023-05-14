from enum import Enum
import cv2
from road_crossing import get_road_crossings_img, try_get_road_crossing_center
import logging
logging.getLogger().setLevel(logging.DEBUG)


class NavigationState(Enum):
    LOCATING_WAYPOINT = 1
    HEAD_TO_WAYPOINT = 2
    CROSSING_WAYPOINT = 3


class RoadCrossingNavigator:

    def __init__(self, input_waypoints_file: str):
        self.waypoints_list = self.get_waypoints_list(input_waypoints_file)
        self.next_waypoint_idx = 0
        self.nav_state = NavigationState.HEAD_TO_WAYPOINT
        self.nav_state_history = [] # push state every 5Hz
        self.tracker = cv2.TrackerCSRT_create()
        self.found_counter = 0
        self.found_counter_threshold = 4
        self.tracking_started = False
        self.down_counter = 0

    def track_road_crossing(self, road_mask):
        """
        Returns road crossing center coordinates if located, None otherwise.
        """
        bbox = None
        found, road_crossing_center = try_get_road_crossing_center(road_mask)
        if found:
            self.found_counter += 1
            self.down_counter = 0
            bbox = [road_crossing_center[0] - 100, road_crossing_center[1] - 100, 100, 100]
        elif self.found_counter > self.found_counter_threshold:
            self.found_counter -= 1
            self.down_counter += 1

        if not found and self.down_counter > 3: # stop tracking
            self.found_counter = 0
            self.tracking_started = False
            del self.tracker
            self.tracker = cv2.TrackerCSRT_create()

        if self.found_counter > self.found_counter_threshold:
            if self.tracking_started:
                ret, bbox = self.tracker.update(road_mask)
            else:
                ret = self.tracker.init(road_mask, bbox)
                self.tracking_started = True
        
        return bbox

    def get_waypoints_list(self, input_waypoints_file: str):
        """
        Returns list of waypoints from input file.
        """
        with input_waypoints_file.open(mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            waypoints_list = [(float(line.split(',')[0]), float(line.split(',')[1])) for line in lines]
        logging.info(f"waypoints list: {waypoints_list}")
        return waypoints_list
    
    def get_navigation_state(self, drone_img_gray):
        """
        Returns the current navigation state of the drone.
        
        Workflow:
        - try locate road crossing
        - if located - start tracking its position with KalmanFilter. if no - nothing changes, keep looking
        - if current position is in close proximity to the next waypoint - switch to CROSSING_WAYPOINT state
        - if current state is crossing waypoint for 2 seconds - switch to HEAD_TO_NEXT_WAYPOINT state, and update next waypoint index, reset kalman filter
        """
        cv2.imshow("road crossings", vis)
        # if not status:
        #     return self.nav_state
        
        if self.nav_state == NavigationState.HEAD_TO_WAYPOINT:
            
            # if located - switch to CROSSING_WAYPOINT state
            # if not - keep heading to next waypoint
            pass
        elif self.nav_state == NavigationState.CROSSING_WAYPOINT:
            # track position of road crossing with kalman filter
            # if current position is in close proximity to the next waypoint - switch to HEAD_TO_NEXT_WAYPOINT state
            # if current state is crossing waypoint for 2 seconds - switch to HEAD_TO_NEXT_WAYPOINT state, and update next waypoint index, reset kalman filter
            pass

        return self.nav_state
    
    def get_next_waypoint(self):
        return self.waypoints_list[self.next_waypoint_idx]
    
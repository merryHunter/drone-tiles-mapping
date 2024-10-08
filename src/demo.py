import argparse
from pathlib import Path
import cv2
from gpslocation import GPSLocationTracker, SatelliteTileManager
import numpy as np
import logging
from logging import DEBUG, INFO
from catalyst.utils import mask_to_overlay_image
from navigation import RoadCrossingNavigator
from road_crossing import RoadCrossingDetector, calculate_distance
from segmentation import RoadSegmentationNetwork
from utils import add_transparent_image
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
VERSION = Path("VERSION").read_text().strip()


def run_experiment(args, output_folder, start_frame=900, end_frame=3000):
    gpstracker = GPSLocationTracker(init_x_position=400, init_y_position=1150, faulty_distance_threshold=150)
    road_segmentor = RoadSegmentationNetwork(args.road_model)
    navigator = RoadCrossingNavigator(args.waypoints_file, found_threshold=2)
    tile_downloader = SatelliteTileManager(args.satellite_tiles_folder)
    waypoint_icon = cv2.imread("media/waypoint-icon.png", cv2.IMREAD_UNCHANGED) 
    waypoint_icon_small = cv2.resize(waypoint_icon, (60, 95))
    waypoint_icon = cv2.resize(waypoint_icon, (200, 250))
    # simulate drone feed
    n_frames = end_frame - start_frame

    tile_cross_tracker_started = False
    tile_load_period = n_frames // 10
    i, j = 0, 1
    tile_copy = tile_downloader.get_tile(0.0, 0.0, index=0)
    for file_path in tqdm(sorted(args.drone_imgs_folder.glob("*.jpeg"))[start_frame:end_frame]):
        drone_img = cv2.imread(str(file_path))
        drone_img_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

        tile = np.copy(tile_copy)
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        if args.algorithm == "kalman_simple":
            current, predicted, distance_from_last = gpstracker.track_position(drone_img_gray, tile_gray)
            x, y = current
            xp, yp = predicted
        elif args.algorithm == "canny":
            x, y = gpstracker.get_current_position_on_tile(drone_img_gray, tile_gray)
        
        road_mask_gray = road_segmentor.predict(drone_img, rotate=False)
        scale_y, scale_x = drone_img.shape[1] / road_mask_gray.shape[1], drone_img.shape[0] / road_mask_gray.shape[0]
        
        crossing_bbox, center_c = navigator.track_road_crossing(road_mask_gray)
        if crossing_bbox is not None:
            logging.debug(f"Found road crossing at {crossing_bbox}")
            crossing_bbox = np.array([int(scale_x * crossing_bbox[0]), int(scale_y * crossing_bbox[1]),
                                      int(scale_x * crossing_bbox[2]),int(scale_y * crossing_bbox[3])])
            
            add_transparent_image(drone_img, waypoint_icon, crossing_bbox[0]+60, crossing_bbox[1]-20)

            # draw road crossing on tile:
            if not tile_cross_tracker_started:
                tile_cross_tracker_started = True
                tile_cx, tile_cy = 410, 620
            logging.debug(f"{tile_cx}, {tile_cy}")
            add_transparent_image(tile, waypoint_icon_small, tile_cx, tile_cy)

        # draw drones position on tile:
        cv2.circle(tile, (x, y), radius=20, color=(0, 0, 255), thickness=-1)
        if args.visualize_raw:
            cv2.putText(tile, str(int(distance_from_last)), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=4)
            cv2.circle(tile, (xp, yp), radius=15, color=(200, 20, 0), thickness=6)
            # todo: overlay road mask over drone image?
            # drone_img = mask_to_overlay_image(image=drone_img, masks=road_mask, threshold=0.4, mask_strength=0.3)
            cv2.circle(drone_img, (crossing_bbox[0]+crossing_bbox[2]//2, crossing_bbox[1]+crossing_bbox[3]//2), radius=15, color=(110, 0, 255), thickness=6)
            cv2.rectangle(drone_img, (crossing_bbox[0], crossing_bbox[1]), (crossing_bbox[0]+crossing_bbox[2], crossing_bbox[1]+crossing_bbox[3]), (0, 255, 0), 2)

        tres = cv2.resize(tile, None, fx=0.65, fy=0.65)
        drone_img = cv2.resize(drone_img, ( 1140, tres.shape[0]))
        stitched_img = np.hstack([drone_img, tres])

        if args.display_only:
            cv2.imshow("Visualize", stitched_img)
            key = cv2.waitKey(20) # pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
        else:
            new_path = output_folder / file_path.name
            cv2.imwrite(str(new_path), stitched_img)

        i += 1
        if i % tile_load_period == 0:
            j += 1
            # mock satellite tile GPS fetcher with pre-downloaded
            tile_copy = tile_downloader.get_tile(0.0, 0.0, index=j)
            if tile_cross_tracker_started:
                tile_cy += 60


if __name__ == "__main__":
    parser.add_argument("--drone_imgs_folder", default=Path.cwd() / "tiles-20-03/dji_0012")
    parser.add_argument("--satellite_tiles_folder", default=Path.cwd() / "tiles-20-03/tilesnew/")
    parser.add_argument("--experiment_name", default="dji0012-allframes")
    parser.add_argument("--algorithm", default="kalman_simple")
    parser.add_argument("--road_model", default=Path.cwd() / "model-22-03.pt")
    parser.add_argument("--waypoints_file", default=Path.cwd() / f"input/waypoints.txt")
    parser.add_argument("--visualize_raw", action="store_true", default=False)
    parser.add_argument("--log_level", default=INFO)
    parser.add_argument("--display_only", action="store_true", default=False, help="If set, only displays the output, does not save image on disk (debug purpose))")
    args = parser.parse_args()
    if args.experiment_name == "":
        output_folder = Path.cwd() / f"results/{VERSION}/"
    else:
        output_folder = Path.cwd() / f"results/{VERSION}_{args.experiment_name}"
    if not output_folder.exists():
        output_folder.mkdir()
    
    run_experiment(args, output_folder)
    
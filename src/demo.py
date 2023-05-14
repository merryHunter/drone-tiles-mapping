import argparse
from pathlib import Path
import cv2
from gpslocation import GPSLocationTracker, SatelliteTileDownloader
import numpy as np
import logging
from logging import DEBUG, INFO
from catalyst.utils import mask_to_overlay_image
from navigation import RoadCrossingNavigator
from road_crossing import calculate_distance, try_get_road_crossing_center
from segmentation import RoadSegmentationNetwork
from utils import add_transparent_image


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
VERSION = Path("VERSION").read_text().strip()


if __name__ == "__main__":
    parser.add_argument("--drone_imgs_folder", default=Path.cwd() / "tiles-20-03/jpeg")
    parser.add_argument("--satellite_tiles_folder", default=Path.cwd() / "tiles-20-03/Tiles/")
    parser.add_argument("--output_folder", default=Path.cwd() / f"results/{VERSION}/")
    parser.add_argument("--algorithm", default="kalman_simple")
    parser.add_argument("--road_model", default=Path.cwd() / "model-22-03.pt")
    parser.add_argument("--waypoints_file", default=Path.cwd() / f"input/waypoints.txt")
    parser.add_argument("--visualize_raw", action="store_true", default=False)
    parser.add_argument("--log_level", default=INFO)
    parser.add_argument("--display_only", action="store_true", default=False, help="If set, only displays the output, does not save image on disk (debug purpose))")
    args = parser.parse_args()
    if not args.output_folder.exists():
        args.output_folder.mkdir()

    tracker = GPSLocationTracker(init_x_position=400, init_y_position=1150, faulty_distance_threshold=150)
    road_segmentor = RoadSegmentationNetwork(args.road_model)
    navigator = RoadCrossingNavigator(args.waypoints_file)
    tile_downloader = SatelliteTileDownloader(args.satellite_tiles_folder)
    waypoint_icon = cv2.imread("media/waypoint-icon.png", cv2.IMREAD_UNCHANGED) 
    # simulate drone feed
    i, j = 0, 1
    tile_copy = tile_downloader.get_tile(0.0, 0.0, index=0)
    for file_path in sorted(args.drone_imgs_folder.glob("*.jpeg"))[20:]:
        drone_img = cv2.imread(str(file_path))
        drone_img_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

        tile = np.copy(tile_copy)
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        if args.algorithm == "kalman_simple":
            current, predicted, metric = tracker.track_position(drone_img_gray, tile_gray)
            x, y = current
            xp, yp = predicted
        elif args.algorithm == "canny":
            x, y = tracker.get_current_position_on_tile(drone_img_gray, tile_gray)
        cv2.putText(tile, str(int(metric)), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=4)
        cv2.circle(tile, (x, y), radius=15, color=(0, 0, 255), thickness=6)
        if args.visualize_raw:
            cv2.circle(tile, (xp, yp), radius=15, color=(200, 20, 0), thickness=6)

        road_mask = road_segmentor.predict(drone_img, rotate=False)
        logging.info(f"{road_mask.shape}, {drone_img.shape}")
        # drone_img = mask_to_overlay_image(image=drone_img, masks=road_mask, threshold=0.4, mask_strength=0.3)
        crossing_bbox = navigator.track_road_crossing(road_mask)
        if crossing_bbox is not None:
            logging.info(f"Found road crossing at {crossing_bbox}")
            scale_y, scale_x = drone_img.shape[1] / road_mask.shape[1], drone_img.shape[0] / road_mask.shape[0]
            crossing_bbox = np.array([int(scale_x * crossing_bbox[0]), int(scale_y * crossing_bbox[1]),
                                      int(scale_x * crossing_bbox[2]),int(scale_y * crossing_bbox[3])])
            # cv2.circle(drone_img, (crossing_bbox[0], crossing_bbox[1]), radius=15, color=(0, 255, 0), thickness=8)
            add_transparent_image(drone_img, waypoint_icon, crossing_bbox[0]+crossing_bbox[2]//2, crossing_bbox[1]+crossing_bbox[3]//2)
        # todo: overlay road mask over drone image

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
            new_path = args.output_folder / file_path.name
            cv2.imwrite(str(new_path), stitched_img)

        i += 1
        if i % 12 == 0:
            j += 2
            # mock satellite tile GPS fetcher with pre-downloaded
            tile_copy = tile_downloader.get_tile(0.0, 0.0, index=j)



            # logging.info(f"Found road crossing at {center}")
            # scale_y, scale_x = drone_img.shape[1] / road_mask.shape[1], drone_img.shape[0] / road_mask.shape[0]
            # center = np.array([int(scale_x * center[0]), int(scale_y * center[1])])
            # logging.info(f"Found road crossing at {center}")
            # road_mask = (road_mask * 255).astype(np.uint8)
            # cv2.circle(drone_img, (center[0],center[1]), radius=15, color=(0, 255, 0), thickness=4)
            # distance_in_pixels = calculate_distance(center, np.array(current))
            # cv2.putText(drone_img, f"{distance_in_pixels:.0f} px", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=4)
        
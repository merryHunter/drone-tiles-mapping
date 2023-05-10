import argparse
from pathlib import Path
import cv2
from gpslocation import GPSLocationTracker
import numpy as np

parser = argparse.ArgumentParser()


if __name__ == "__main__":
    parser.add_argument("--drone_imgs_folder", default=Path.cwd() / "tiles-20-03/jpeg")
    parser.add_argument("--satellite_tiles_folder", default=Path.cwd() / "tiles-20-03/Tiles/")
    parser.add_argument("--output_folder", default=Path.cwd() / "results/0.1/")
    parser.add_argument("--display_only", action="store_true", default=False, help="If set, only displays the output, does not save image on disk (debug purpose))")
    args = parser.parse_args()
    if not args.output_folder.exists():
        args.output_folder.mkdir()

    tracker = GPSLocationTracker(0,0,0)
    # simulate drone feed
    i, j = 0, 1
    for file_path in sorted(args.drone_imgs_folder.glob("*.jpeg")):
        drone_img = cv2.imread(str(file_path))
        drone_img_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

        # mock satellite tile GPS fetcher with pre-downloaded
        fpath = args.satellite_tiles_folder / f"{j}.png"
        tile = cv2.imread(str(fpath))
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        
        x, y = tracker.get_current_position_on_tile(drone_img_gray, tile_gray)

        cv2.circle(tile, (x, y), radius=15, color=(0, 0, 255), thickness=6)
        
        tres =  cv2.resize(tile, None, fx=0.65, fy=0.65)
        drone_img = cv2.resize(drone_img, ( 1040, tres.shape[0]))
        stitched_img = np.hstack([drone_img, tres])
        # print(stitched_img.shape)
        if args.display_only:
            cv2.imshow("Visualize", stitched_img)
            key = cv2.waitKey(20)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
        else:
            new_path = args.output_folder / file_path.name
            cv2.imwrite(str(new_path), stitched_img)

        i += 1
        if i % 12 == 0:
            j += 2

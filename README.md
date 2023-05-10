# Drone2tiles mapping

As part of work, road semantic segmentation has been developed.

# Issues

- what to do during take off?

# todo:
- visualize crossings (and mark them as "passed")
- convert points to real GPS locations
- fuse segmentation input
- load gps tile automatically
- take into account direction change as optical flow for filtering and gps loading
- display distance between kalman filtered and raw predictions as moving average over last 5 predictions

# Release history 

## Version 0.0.2

Added Kalman filter, faulty prediction filtering based on distance threshold.

## Version 0.0.1

Canny edge preprocessing, scale invariant template matching, each frame independently.


## ffmpeg commands

Create video from images:

ffmpeg -framerate 30 -pattern_type glob -i '*.jpeg' -c:v libx264 -pix_fmt yuv420p out.mp4

If error height is not divisible by two -  add `-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"`.


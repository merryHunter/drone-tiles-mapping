# Drone2tiles mapping

As part of work, road semantic segmentation has been developed.

# Issues

- what to do during take off?

# todo:
- test on more videos
- investigate problem with higher frame video input
- improve template matching of drone position
- add forward intertial kinematics to kalman fitler to take into account velocity (add velocity to state vector)
- convert points to real GPS locations
- fetch gps tile automatically based on current location
- create tool for gps tile download


# Release history 

## Version 0.0.5

Added hard-coded visualization for road crossing on satellite tile. Must be removed and replaced with automatic tile fetcher and gps to image coordinates translation.

## Version 0.0.4

Improved road crossing detection and visualisation by combining results from contour recognition and hough lines intersection.

## Version 0.0.3

Added tracking of road crossing on drone view

## Version 0.0.2

Added Kalman filter, faulty prediction filtering based on distance threshold.

## Version 0.0.1

Canny edge preprocessing, scale invariant template matching, each frame independently.


## ffmpeg commands

Create video from images:

ffmpeg -framerate 30 -pattern_type glob -i '*.jpeg' -c:v libx264 -pix_fmt yuv420p out.mp4

If error height is not divisible by two -  add `-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"`.

`ffmpeg -framerate 10 -pattern_type glob -i '*.jpeg' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ../videos/0.0.3.mp4`

`ffmpeg -i ../DJI_0012.MP4 -vf fps=30 out%d.jpeg`
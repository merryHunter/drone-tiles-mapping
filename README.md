# Drone2tiles mapping

As part of work, road semantic segmentation has been developed.

# Issues

- what to do during take off?

# todo:
- mark road crossing as "passed"
- add road crossing on satellite view
- convert points to real GPS locations
- load gps tile automatically


# Release history 


## Version 0.0.3

Added tracking road of road crossing on drone view

## Version 0.0.2

Added Kalman filter, faulty prediction filtering based on distance threshold.

## Version 0.0.1

Canny edge preprocessing, scale invariant template matching, each frame independently.


## ffmpeg commands

Create video from images:

ffmpeg -framerate 30 -pattern_type glob -i '*.jpeg' -c:v libx264 -pix_fmt yuv420p out.mp4

If error height is not divisible by two -  add `-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"`.

`ffmpeg -framerate 10 -pattern_type glob -i '*.jpeg' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ../videos/0.0.3.mp4`
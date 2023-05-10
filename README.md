# Drone2tiles mapping

As part of work, road semantic segmentation has been developed.


# Version 0.1

Canny edge preprocessing, scale invariant template matching, each frame independently.


## ffmpeg commands

Create video from images:

ffmpeg -framerate 30 -pattern_type glob -i '*.jpeg' -c:v libx264 -pix_fmt yuv420p out.mp4

If error height is not divisible by two -  add `-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"`.


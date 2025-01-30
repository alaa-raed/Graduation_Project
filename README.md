# Drowsiness and Yawn detection with voice alert using Dlib

Simple code in python to detect Drowsiness and Yawn and alert the user using Dlib.

## Dependencies

1. Python 3
2. opencv
3. dlib
4. imutils
5. scipy
6. numpy
7. argparse

## Run 

```
Python3 drowsiness_yawn.py -- webcam 0		//For external webcam, use the webcam number accordingly
```

## Setups

Change the threshold values according to your need
```
EYE_AR_THRESH = 0.4
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20`	//change this according to the distance from the camera
```


## Acknowledgments

* https://www.pyimagesearch.com/



* To install podman on linux 
```
sudo apt install -y podman
```

* To create image of the project
```
podman build -t drowsinessv1 .
```

* To run the container
```
podman run -it --rm   --device=/dev/video0   -v /tmp/.X11-unix:/tmp/.X11-unix   -e DISPLAY=$DISPLAY   --security-opt label=disable   drowsinessv1
```

#!/usr/bin/env python
import os

if __name__=="__main__":
    #cmd = "nvidia-docker run -it -p 8080:8080 -p 8888:8888 -v %s:/host priya-kaolin" % (os.path.join(os.getcwd(), '..'))
    cmd = "xhost +local:root && \
         nvidia-docker run \
         -p 8080:8080 -p 8888:8888 \
         -v /tmp/.X11-unix:/tmp/.X11-unix \
         --gpus all \
         -e DISPLAY=$DISPLAY \
         -e QT_X11_NO_MITSHM=1 \
         -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
         -v %s:/host \
         -it priya-kaolin" % (os.path.join(os.getcwd(), '..'))
    code = os.system(cmd)

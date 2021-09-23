#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "nvidia-docker build -t priya-kaolin . "
    code = os.system(cmd)

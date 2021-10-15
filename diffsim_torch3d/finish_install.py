#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    os.system('g++ -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -std=c++14 -fPIC -I./arcsim/src/ -I./arcsim/dependencies/include -I/root/miniconda3/envs/kaolin/lib/python3.6/site-packages/torch/include -I/root/miniconda3/envs/kaolin/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/root/miniconda3/envs/kaolin/lib/python3.6/site-packages/torch/include/TH -I/root/miniconda3/envs/kaolin/lib/python3.6/site-packages/torch/include/THC -I/root/miniconda3/envs/kaolin/include/python3.6m -c pybind/bind.cpp -o build/temp.linux-x86_64-3.6/pybind/bind.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=arcsim -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14')

    if not os.path.exists('build/lib.linux-x86_64-3.6'):
        os.system('mkdir build/lib.linux-x86_64-3.6')
    
    os.system('g++ -pthread -shared -B /root/miniconda3/envs/kaolin/compiler_compat -L/root/miniconda3/envs/kaolin/lib -Wl,-rpath=/root/miniconda3/envs/kaolin/lib -Wl,--no-as-needed -Wl,--sysroot=/ -std=c++14 build/temp.linux-x86_64-3.6/pybind/bind.o -Lobjs -L./arcsim/dependencies/lib -lmake_pytorch -ljson -ltaucs -lalglib -lpng -lz -llapack -lblas -lboost_system -lboost_filesystem -lboost_thread -lgomp -lglut -lGLU -lGL -o build/lib.linux-x86_64-3.6/arcsim.cpython-36m-x86_64-linux-gnu.so')

#!/bin/bash

conda init bash && source ~/.bashrc && conda activate kaolin && python setup.py develop

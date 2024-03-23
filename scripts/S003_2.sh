#!/bin/bash
echo "run c017"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c017.mp4 --reid-weights net_11.pth --name c017
echo "run c018"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c018.mp4 --reid-weights net_11.pth --name c018
echo "run c019"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c019.mp4 --reid-weights net_11.pth --name c019


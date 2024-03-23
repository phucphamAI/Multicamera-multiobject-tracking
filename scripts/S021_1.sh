#!/bin/bash
echo "run c118"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c118.mp4 --reid-weights net_11.pth --name c118 
echo "run c119"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c119.mp4 --reid-weights net_11.pth --name c119 
echo "run c120"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c120.mp4 --reid-weights net_11.pth --name c120 
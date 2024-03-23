#!/bin/bash
echo "run c124"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c124.mp4 --reid-weights net_11.pth --name c124 
echo "run c125"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c125.mp4 --reid-weights net_11.pth --name c125 
echo "run c126"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c126.mp4 --reid-weights net_11.pth --name c126 
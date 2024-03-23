#!/bin/bash
echo "run c121"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c121.mp4 --reid-weights net_11.pth --name c121  
echo "run c122"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c122.mp4 --reid-weights net_11.pth --name c122
echo "run c123"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c123.mp4 --reid-weights net_11.pth --name c123  
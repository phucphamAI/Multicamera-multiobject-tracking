#!/bin/bash
echo "run c079"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c079.mp4 --reid-weights net_11.pth --name c079
echo "run c080"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c080.mp4 --reid-weights net_11.pth --name c080
echo "run c081"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c081.mp4 --reid-weights net_11.pth --name c081


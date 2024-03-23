#!/bin/bash
echo "run c014"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c014.mp4 --reid-weights net_11.pth --name c014
echo "run c015"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c015.mp4 --reid-weights net_11.pth --name c015
echo "run c016"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c016.mp4 --reid-weights net_11.pth --name c016


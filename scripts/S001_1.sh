#!/bin/bash
echo "run c001"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c001.mp4 --reid-weights net_11.pth --name C001
echo "run c002"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c002.mp4 --reid-weights net_11.pth --name C002
echo "run c003"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c003.mp4 --reid-weights net_11.pth --name C003
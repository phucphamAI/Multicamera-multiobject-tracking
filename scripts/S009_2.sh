#!/bin/bash
echo "run c050"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c050.mp4 --reid-weights net_11.pth --name c050
echo "run c051"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c051.mp4 --reid-weights net_11.pth --name c051
echo "run c052"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c052.mp4 --reid-weights net_11.pth --name c052


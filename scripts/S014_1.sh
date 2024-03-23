#!/bin/bash
echo "run c076"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c076.mp4 --reid-weights net_11.pth --name c076
echo "run c077"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c077.mp4 --reid-weights net_11.pth --name c077
echo "run c078"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c078.mp4 --reid-weights net_11.pth --name c078


#!/bin/bash
echo "run c127"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c127.mp4 --reid-weights net_11.pth --name c127  
echo "run c128"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c128.mp4 --reid-weights net_11.pth --name c128  
echo "run c129"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c129.mp4 --reid-weights net_11.pth --name c129 
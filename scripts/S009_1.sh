#!/bin/bash
echo "run c047"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c047.mp4 --reid-weights net_11.pth --name c047
echo "run c048"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c048.mp4 --reid-weights net_11.pth --name c048
echo "run c049"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c049.mp4 --reid-weights net_11.pth --name c049


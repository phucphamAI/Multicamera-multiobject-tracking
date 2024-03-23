#!/bin/bash
echo "run c004"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c004.mp4 --reid-weights net_11.pth --name c004
echo "run c005"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c005.mp4 --reid-weights net_11.pth --name C005
echo "run c006"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c006.mp4 --reid-weights net_11.pth --name C006
echo "run c007"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c007.mp4 --reid-weights net_11.pth --name C007
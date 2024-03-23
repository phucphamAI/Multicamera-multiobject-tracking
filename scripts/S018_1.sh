#!/bin/bash
echo "run c100"
python3 yolo/v8/detect/pipeline.py --source /media/aivn24/partition1/HungAn/Track1/dataset/test/c100.mp4 --reid-weights net_11.pth --name c100 
echo "run c101"
python3 yolo/v8/detect/pipeline.py --source /media/aivn24/partition1/HungAn/Track1/dataset/test/c101.mp4 --reid-weights net_11.pth --name c101 
echo "run c102"
python3 yolo/v8/detect/pipeline.py --source /media/aivn24/partition1/HungAn/Track1/dataset/test/c102.mp4 --reid-weights net_11.pth --name c102 
#!/bin/bash
echo "run c103"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c103.mp4 --reid-weights net_11.pth --name c103 
echo "run c104"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c104.mp4 --reid-weights net_11.pth --name c104 
echo "run c105"
python3 yolo/v8/detect/pipeline.py --source ../Track1/test/c105.mp4 --reid-weights net_11.pth --name c105 
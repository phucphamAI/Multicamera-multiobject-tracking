### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

### Train
Train a model by
```bash
python train.py --gpu_ids 0 --name swin --train_all --batchsize 32  --data_dir ../Dataset --use_swin
```

### Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name swin --test_dir your_data_path  --batchsize 32 --which_epoch 59
```

### Extract
```
python extract.py --name swin --video_path ../results_OD/c001.mp4 --result_path ../results_OD/c001.txt --pkl_path ../results_OD/c001.pkl
```
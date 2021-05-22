### Kaggle-Bengali
---
#### EDA
- Multi-class 분포 
  - 분포 -> image (시각적으로 한눈에 볼 수 있게 하면 좋아요)
  - Image size (Max/Min/Avg/Max_width/Max_height/mean) -> 사이즈 다를 수 있으니까 checkcheck
  - image-pixel mean variance (normalize 할 때)
  - 흑백 or 컬러 
 
- parquet 포맷 열어보기 (구성)

#### Model code 틀 만들기 

### Command
```python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/model/resnet50x1.yaml configs/dataset/cifar10.yaml configs/optimizer/adamw.yaml configs/scheduler/multistep.yaml configs/loss/ce.yaml --use_amp```

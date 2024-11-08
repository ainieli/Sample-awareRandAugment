# Sample-aware RandAugment

## Requirements

- pytorch 1.13.1

- timm 0.6.7

- numpy 1.21.5

## Training

- CIFAR-10/100
  
  - Change the model and training hyperparameters in SRA_CIFAR.py
  
  - Run `python SRA_CIFAR.py`

- ImageNet
  
  - Set DDP hyperparameters (NUM_NODES) and assign GPU device ID in SRA_ImageNet.py
    
    - check parameter "num_gpu" == NUM_NODES to ensure correct batchsize in DDP
  
  - Change the model and training hyperparameters in SRA_ImageNet.py
  
  - Run `python -m torch.distributed.run --nproc_per_node NUM_NODES --master_port PORT_ID SRA_ImageNet.py`

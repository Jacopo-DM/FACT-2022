

# FACT 2022
# SCOUTER: Slot Attention-based Classifier for Explainable Image Recognition 



## Training

### Imagenet

For training ImageNet for 100 categories, a high-RAM GPU is needed. We used Google Cloud
to train the ImageNet models.

##### Pretrain the FC ResNest26 backbone for 100 categories

```bash
python train.py --dataset ImageNet --model resnest26d --batch_size 70 --epochs 20 \
--num_classes 100 --use_slot false --vis false --channel 2048 --freeze_layers 0 \
--dataset_dir data/imagenet/ILSVRC/Data/CLS-LOC/
```

##### Train the positive Scouter for 100 categories with lambda 10

```bash
python train.py --dataset ImageNet --model resnest26d --batch_size 70 --epochs 20 \
--num_classes 100 --use_slot true --use_pre false --loss_status 1 --slots_per_class 1 --output_dir lambda_3/ \
--power 2 --num_workers 4 --to_k_layer 3 --lambda_value 3 --vis false --channel 2048 --freeze_layers 0 \
--dataset_dir data/imagenet/ILSVRC/Data/CLS-LOC/
```

##### Train the negative Scouter for 100 categories with lambda 10

```bash
python train.py --dataset ImageNet --model resnest26d --batch_size 70 --epochs 20 \
--num_classes 10 --use_slot true --use_pre false --loss_status -1 --slots_per_class 1 \
--power 2 --to_k_layer 3 --lambda_value 1 --vis false --channel 2048 --freeze_layers 0 \
--dataset_dir ../data/imagenet/ILSVRC/Data/CLS-LOC/
```

###### You can enable distributed training using the following arguments in your command:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --world_size 4
```



### CUB-200 Dataset

##### Pre-training FC ResNest50 backbone (50 categories)
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --world_size 1 
--dataset CUB200 --model resnest50d --num_workers 0 --batch_size 16 --epochs 150 \
--num_classes 50 --use_slot false --vis false --channel 2048 \
--dataset_dir data/CUB_200_2011
```

##### Pre-training FC ResNest26 backbone (100 categories)

```bash
python train.py --dataset CUB200 --model resnest26d --batch_size 64 --epochs 150 \
--num_classes 100 --use_slot false --vis false --channel 2048 --num_workers 4 \
--dataset_dir data/CUB200/CUB_200_2011
```

##### Positive Scouter on CUB-200 (50 categories)

```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --world_size 1 --dataset CUB200 \
--model resnest50d --batch_size 16 --epochs 150 \
--num_classes 50 --num_workers 2 --use_slot true --use_pre true --loss_status 1 --slots_per_class 5 \
--power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 \
--dataset_dir data/CUB_200_2011/
```

##### Negative Scouter on CUB-200 (50 categories)

```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --world_size 1 --dataset CUB200 \
--num_workers 2 --model resnest50d --batch_size 16 --epochs 150 \
--num_classes 50 --use_slot true --use_pre true --loss_status -1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 \
--dataset_dir data/CUB_200_2011
```

The CUB-200 experiments with different number of categories has been trained with
similar commands by just adjusting the number of classes and/or reducing the number of workers for memory issues.

###Generate results and visualization

The results (metrics) and visualizations are produced in the ```.ipynb``` file in this repository. 
If you don't have a GPU, 
we strongly recommend to run this notebook on Google Colab (Pro version even better),
since a lot of computational power is needed for calculating the area size, precision, IAUC, DAUC and sensitivity metrics.

Instructions on how to download the datasets and model files are in the notebook file.

## Acknowledgements
We would like to thank SurfSara for providing us their computational resources.
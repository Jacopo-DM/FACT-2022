

# FACT 2022
## SCOUTER: Slot Attention-based Classifier for Explainable Image Recognition 


### Original codebase from https://github.com/wbw520/scouter
#### Python files changed in codebase 
```
engine.py (significantly)
train.py
test.py
dataset\ConText.py              (to retrieve the height and width of Imagenet images)
sloter\slot_model.py            (to get the attention map)
sloter\utils\slot_attention.py  (to get the attention map)
```
#### New python files
```
get_results.py
restruct_imgnet.py   (to restructure the ILSVRC ImageNet dataset)
```

#### Imported python files from other papers (https://arxiv.org/abs/1806.07421, https://arxiv.org/abs/1901.09392)
```
IAUC_DAUC_eval.py       (to get the IAUC and DAUC scores)
IAUC_DAUC_eval_utils.py
infid_sen_utils.py      (to get sensitivity)
```


#### If there's any questions about visualization or metrics, feel free to e-mail bartvanvulpen@icloud.com


## Training
NOTE: Training a model can take up to several hours.
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
--power 2 --num_workers 4 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 0 \
--dataset_dir data/imagenet/ILSVRC/Data/CLS-LOC/
```

##### Train the negative Scouter for 100 categories with lambda 10

```bash
python train.py --dataset ImageNet --model resnest26d --batch_size 70 --epochs 20 \
--num_classes 10 --use_slot true --use_pre false --loss_status -1 --slots_per_class 1 \
--power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 0 \
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

### Generate results and visualization

The results (metrics) and visualizations are produced in the ```.ipynb``` file in this repository. 
If you don't have a GPU, 
we strongly recommend to run this notebook on Google Colab (Pro version even better),
since a lot of computational power is needed for calculating the area size, precision, IAUC, DAUC and sensitivity metrics.

Instructions on how to download the datasets and model files are in the notebook file.

## Acknowledgements
We would like to thank SurfSara for providing us their computational resources.
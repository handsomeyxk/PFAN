# PFAN
### PFAN_SR
>This project is based on BasicSR.

### Requirements:
>Python >= 3.7
>PyTorch >= 1.7
>BasicSR >= 1.3.5

### Datasets:
>Training: DIV2K or DF2K.
>Testing: Set5, Set14, BSD100, Urban100, Manga109 (Google Drive/Baidu Netdisk).
>Preparing: Please refer to the Dataset Preparation of BasicSR.

### Train and Test：
>The BasicSR framework is utilized to train our PFAN, also testing.

Training with the example option
# Single GPU Training

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/PFAN/train_PFAN_x4.yml

# Distributed Training

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/PFAN/train_PFAN_x4.yml --launcher pytorch

Testing with the example option
# Single GPU Testing

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/PFAN/test_PFAN_x4.yml

# Distributed Testing

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/PFAN/test_PFAN_x4.yml --launcher pytorch


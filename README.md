# HSVL-BERT

## Introduction
Decrease the range of answer candidates and extract effective embeddings is required for visual question answering. 
we proposed HSVL-BERT a consist of a transformer with hierarchical syntactic and a speech act identification module for classifying question type.
![](./figs/dataflow.PNG)

![](./figs/VLBERT-v2.PNG)

### Data

See [PREPARE_DATA.md](data/PREPARE_DATA.md).

### Pre-trained Models

See [PREPARE_PRETRAINED_MODELS.md](model/pretrained_model/PREPARE_PRETRAINED_MODELS.md).

## Training
```
./scripts/nondist_run.sh <num_gpus> <task>/train_end2end.py <path_to_cfg> <dir_to_store_checkpoint>
```
* ```<num_gpus>```: number of gpus to use.
* ```<task>```: pretrain/vqa.
* ```<path_to_cfg>```: config yaml file under ```./cfgs/<task>```.
* ```<dir_to_store_checkpoint>```: root directory to store checkpoints.
  
Following is a more concrete example:
```
./scripts/nondist_run.sh pretrain/train_end2end.py ./cfgs/pretrain/base_prec_4x16G_fp32.yaml
```

## Evaluation
### VQA
* Generate prediction results on test set for [EvalAI submission](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview):
  ```
  python vqa/test.py \
    --cfg <cfg_file> \
    --ckpt <checkpoint> \
    --gpus <indexes_of_gpus_to_use> \
    --result-path <dir_to_save_result> --result-name <result_file_name>
  ```

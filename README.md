# Group-MATES: Group-Level Data Selection for Efficient Pretraining

This is the code for [Group-MATES: Group-Level Data Selection for Efficient Pretraining](https://arxiv.org/abs/2502.14709).
The implementation is mainly based on [DCLM](https://github.com/mlfoundations/dclm) and [MATES](https://github.com/cxcscmu/MATES).

## Quick Links

- [Prerequisites](#prerequisites)
- [Get Data Pool](#get-data-pool)
- [Group-MATES Selection](#group-mates-selection)
    - [Step1: Collect Oracle Data Influences (Random Rollout Policy)](#step1-collect-oracle-data-influences-random-rollout-policy)
    - [Step2: Train Initial Relational Data Influence Model](#step2-train-initial-relational-data-influence-model)
    - [Step3: Collect Oracle Data Influences (Bootstrap Rollout Policy)](#step3-collect-oracle-data-influences-bootstrap-rollout-policy)
    - [Step4: Train Final Relational Data Influence Model](#step4-train-final-relational-data-influence-model)
    - [Step5: Cluster-Based Efficient Inference](#step5-cluster-based-efficient-inference)
- [Tokenization](#tokenization)
- [Pretraining](#pretraining)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)


## Prerequisites

The code is tested on Python 3.10.16. Install basic dependencies:

```bash
pip install -r requirements.txt
```

Run setup.py to download necessary files:

```bash
python setup.py install
```

Install AWS CLI and configure it with your AWS credentials:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

aws configure
aws s3 ls --summarize --human-readable --recursive s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_0_of_10/
```

## Get Data Pool

We start from the DCLM-refinedweb pool and process the data with DCLM-fastText ourselves. Please change the file location based on your system.

See `scripts/fasttext.sh`:

- `source_ref_paths`: one data split on AWS ("dataset_url": "s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_0_of_10"), we can simply use global-shard_01_of_10/local-shard_x_of_10 for more splits
- `output_dir`: processed text data dir

One processed split will have ~36B tokens (60GB - 70GB in size), so if the scale requires 138B tokens (e.g., 7B-1x), we need at least 8 splits to be processed since our later selection ratio will be ~0.5.

## Group-MATES Selection

### Step1: Collect Oracle Data Influences (Random Rollout Policy)

Generate the random rollout data used for oracle data influence collection:

```bash
python group-mates/oracle/tokenize_01_0.py
```

- Modify `input_dir` to your tokenized data dir

Run `scripts/probe.sh`:

- `CKPT`: the checkpoint name used for oracle data influence collection
- `scale`: DCLM running scale, please find the supported ones in `training/configs`
- `data-config`: no need to change
- `logs`: where to store the checkpoint

### Step2: Train Initial Relational Data Influence Model

```bash
export CKPT=YOUR_CHECKPOINT_NAME
torchrun --nproc-per-node 8 group-mates/modeling/train_data_influence_model.py init
```

### Step3: Collect Oracle Data Influences (Bootstrap Rollout Policy)

Generate the bootstrap rollout data used for oracle data influence collection:

```bash
export CKPT=YOUR_CHECKPOINT_NAME
python group-mates/modeling/bootstrap.py
```

Run `scripts/probe.sh` again. It will automatically detect and use the bootstrap data.

### Step4: Train Final Relational Data Influence Model

```bash
export CKPT=YOUR_CHECKPOINT_NAME
torchrun --nproc-per-node 8 group-mates/modeling/train_data_influence_model.py final
```

### Step5: Cluster-Based Efficient Inference

Run `group-mates/inference/bert_tokenize.py`:

- Modify `data_dir` to your processed text data dir
- Modify `output_dir` to your bert tokenized data dir

You can split the data into multiple shards to speed up the tokenization by:

```bash
index=0
for s in {0..7}; do
    echo $s
    nohup python group-mates/inference/bert_tokenize.py --shard $s 8 > log_job_s${s}.out 2>&1 &
    ((index=(index+1)%8))
done
```

Run `group-mates/modeling/predict_data_influence.py`:

- Modify `data_dir` to your bert tokenized data dir
- Modify `model_dir` to your data influence model dir
- Modify `output_dir` to your prediction dir

You can split the data into multiple shards to speed up the prediction by:

```bash
index=0
for s in {0..7}; do
    echo $s
    export CKPT=YOUR_CHECKPOINT_NAME
    CUDA_VISIBLE_DEVICES=$index nohup python group-mates/modeling/predict_data_influence.py --shard $s 8 > log_job_s${s}.out 2>&1 &
    ((index=(index+1)%8))
done
```

Modify all paths in `group-mates/inference/clustering/configs/group-mates.yaml` with your prediction dir and run:

```bash
cd group-mates/inference/clustering
export PYTHONPATH=$(pwd)
cd ..
python clustering.py
python select_indices.py
```

The selected indices will be in your prediction dir.

Run `group-mates/tokenization/select_data_with_indices.py`:

- Modify `args.output_dir` to your prediction dir (it will find the selected indices automatically)
- Modify `data_dir` to your bert tokenized data dir
- Modify `file_dir` to your processed text data dir

The final selected data will be in the `{args.output_dir}/processed_data`.

## Tokenization

Please install rust in your conda environment.

Run `rust_processing/tokshuf-rs/rust_tokenize.sh`:

- `input`: the original text data dir `{args.output_dir}/processed_data`
- `output`: the tokenized data dir

## Pretraining

Run `scripts/pretrain.sh`:

- `scale`: DCLM running scale, please find the supported ones in `training/configs`
- `data-config`: specify the run name ("name") and the tokenized data location ("manifest_url"), create one when you have a new dataset
- `logs`: where to store the checkpoint
- `multiple-data-passes`: used to allow multiple epochs

## Evaluation

Run `scripts/eval.sh`:

- `method`: the generated checkpoint dir name
- `checkpoint`: the specific epoch you want to evaluate
- `model`: model scale config in `training/open_lm_configs`
- `output-file`: where to store the evaluation result

## Citation

Please cite our paper if you use Group-MATES in your work:

```bibtex
@article{yu2025group,
   title={Group-Level Data Selection for Efficient Pretraining},
   author={Yu, Zichun and Peng, Fei and Lei, Jie and Overwijk, Arnold and Yih, Wen-tau and Xiong, Chenyan},
   journal = {ArXiv preprint},
   year={2025}
}
```

## License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](http://creativecommons.org/licenses/by-nc/4.0/).

#!/bin/bash
set -x

merge_train_path="data/anli_ft_correct_merge_train_shuffled_shuffled.parquet"
merge_test_path="data/anli_ft_correct_merge_test_shuffled_shuffled.parquet"
project_name=verl_sft_correct_merge

train_files=$merge_train_path
test_files=$merge_test_path

MODEL_PATH=/data/app/yangyahe/base_model/Qwen-Qwen3-4B-Instruct-2507

save_path=checkpoints/$project_name

CUDA_VISIBLE_DEVICES="2"
nproc_per_node=1
lr=1e-5

messages_key=prompt

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.multiturn.enable=true \
    data.multiturn.messages_key=$messages_key \
    optim.lr=$lr \
    data.micro_batch_size=4 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$(basename "$MODEL_PATH") \
    trainer.logger='["console", "tensorboard"]' \
    trainer.total_training_steps=1 $@ \
    ulysses_sequence_parallel_size=$nproc_per_node \
    use_remove_padding=true

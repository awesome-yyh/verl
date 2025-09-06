set -x

# 包含正确句的比例: 394621 / 787030 = 0.50, 正确到正确的比例: 248742 / 787030 = 0.32
# 包含正确句的比例: 502 / 968 = 0.52, 正确到正确的比例: 322 / 968 = 0.33

# merge_train_path="data/anli_ft_train_system_tgt_add_src.parquet"
# merge_test_path="data/anli_ft_test_system_tgt_add_src.parquet"
# project_name=verl_grpo_merge

# merge_train_path="../term-corrector/data/anli_ft_correct_merge_train_shuffled_shuffled.parquet"
# merge_train_path="../term-corrector/data/www_merge_拦截_correct_merge_train_anli_ft_deduplicated_shuffled.parquet"
# merge_test_path="../term-corrector/data/anli_ft_correct_merge_test_shuffled_shuffled.parquet"
merge_train_path="data/anli_ft_lanjie_merge_x_train_deduplicated_shuffled_all2x_deduplicated_shuffled_error_correct_noop.parquet"
merge_test_path="data/anli_ft_lanjie_merge_x_test_deduplicated_shuffled_all2x_deduplicated_shuffled_error_correct_noop.parquet"
project_name=verl_grpo_correct_merge2e6_error_correct_noop

# merge_train_path="data/anli_ft_correct_train.parquet"
# merge_test_path="data/anli_ft_correct_test.parquet"
# project_name=verl_grpo_correct

# merge_train_path=$merge_test_path  # 临时测试

train_files="['$merge_train_path']"
test_files="['$merge_test_path']"

# MODEL_PATH=../base_model/Qwen-Qwen2.5-1.5B-Instruct
MODEL_PATH=../base_model/Qwen-Qwen2.5-3B-Instruct
# MODEL_PATH=checkpoints/verl_grpo_merge/Qwen-Qwen2.5-3B-Instruct/global_step_1548/actor-hf_model
MODEL_PATH=../base_model/Qwen-Qwen3-4B-Instruct-2507
# MODEL_PATH=hf_newtoken # 同上，grpo558之后+sft
# MODEL_PATH=hf_modelnewtoken_justsft  # 同上，直接sft
# MODEL_PATH=../term-corrector/checkpoints/verl_grpo_correct_merge/Qwen-Qwen2.5-3B-Instruct/global_step_558-actor-hf

# 太大会oom
# merge_test_path
# 4/4/4: Training Progress:   1%|          | 59/11160 [07:24<59:40:15, 19.35s/it]
# 32/4/4: Training Progress:   3%|▎         | 44/1395 [05:14<28:40:40, 76.42s/it]
# 32/32/4: Training Progress:   3%|▎         | 44/1395 [05:16<29:00:44, 77.31s/it]
# 32/32/32: Training Progress:   4%|▎         | 49/1395 [05:02<11:23:38, 30.47s/it]
# 1024/32/32: 几分钟直接训练完...

train_batch_size=128  # train_batch_size * rollout_n 即为每次rollout（采样）阶段收集的总样本数量
ppo_mini_batch_size=64  # 采样的响应被划分的batch，用于更新actor，是所有worker的全局大小
micro_batch_size=32 # 每个GPU实际一次处理的样本数, 用于将mini_batch进一步划分，以适应GPU内存限制, 如果出现OOM，优先减小micro_batch_size
gpu_memory_utilization=0.75
n_gpus_per_node=4
CUDA_VISIBLE_DEVICES="0,1,2,3"
save_freq=80

# 太大会训练不稳定
lr=2e-6
kl_loss_coef=0.001
rollout_n=8  # grpo的群组大小

# use_kl_loss 对于grpo需要设置为true, DrGRPO需要设置为false

# 提高内存使用阈值（默认约95%，可调整为98%，范围0-100）
export RAY_memory_usage_threshold=0.98
export RAY_object_store_memory_buffer=0.3
export RAY_memory_monitor_refresh_ms=0

RAY_RESOURCES="{GPU: $n_gpus_per_node}" \
RAY_RAYLET_GPU_DEVICES=$CUDA_VISIBLE_DEVICES \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
HYDRA_FULL_ERROR=1 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$(basename "$MODEL_PATH") \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=20 \
    trainer.total_epochs=15 $@

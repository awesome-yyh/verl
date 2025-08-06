set -x

# 包含正确句的比例: 394621 / 787030 = 0.50, 正确到正确的比例: 248742 / 787030 = 0.32
# 包含正确句的比例: 502 / 968 = 0.52, 正确到正确的比例: 322 / 968 = 0.33

merge_train_path="data/anli_ft_train_system_tgt_add_src.parquet"
merge_test_path="data/anli_ft_test_system_tgt_add_src.parquet"

merge_train_path=$merge_test_path  # 临时测试

train_files="['$merge_train_path']"
test_files="['$merge_test_path']"

MODEL_PATH=/data/app/yangyahe/base_model/Qwen-Qwen2.5-7B-Instruct

# 太大会oom
train_batch_size=32  # 每次rollout（采样）阶段收集的总样本数量
ppo_mini_batch_size=32  # 11 每次梯度更新使用的样本数, 通常是train_batch_size的一个子集
micro_batch_size=4 # 1每个GPU实际一次处理的样本数, 用于将mini_batch进一步划分，以适应GPU内存限制, 如果出现OOM，优先减小micro_batch_size
gpu_memory_utilization=0.5

# 太大会训练不稳定
lr=1e-6

RAY_RESOURCES='{"GPU": 2}' \
RAY_RAYLET_GPU_DEVICES="6,7" \
CUDA_VISIBLE_DEVICES="6,7" \
HYDRA_FULL_ERROR=1 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name='verl_ppo_merge' \
    trainer.experiment_name=$(basename "$MODEL_PATH") \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=1 \
    trainer.use_legacy_worker_impl=auto \
    trainer.total_epochs=15 $@

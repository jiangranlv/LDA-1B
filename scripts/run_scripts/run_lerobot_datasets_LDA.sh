export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export WANDB_API_KEY=wandb/api/key # replace with your wandb api key
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)


Framework_name=QwenMMDiT
base_vlm=/path/to/pretrained/VLM
vision_encoder_path=/path/to/pretrained/vision/encoder # should be the parent path of vision encoder ckpt

freeze_module_list='qwen_vl_interface, action_model.vision_encoder' # if you would like to directly train on the robocasa dataset, unfreeze vlm could obtain better performance
DIT_TYPE="DiT-L"

llavadata="asv2_conversation_en,asv2_detailed_description_en"
data_root_dir=/path/to/data/directory
data_mix=data_mix_name # should be recorded in data_config.py

obs_horizon=1
state_dim=12
action_dim=12
max_num_embodiments=32
use_delta_action=true
positional_embeddings=null # null, sinusoidal, rope

repeated_diffusion_steps=1

future_obs_index=5
run_root_dir=/path/to/save/training/results # replace with your own path
run_id=/run/id

pretrained_checkpoint=null # set to null if training from scratch

only_policy=false

export WANDB_MODE=disabled
wandb_entity=your/wandb/entity

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  lda/training/train_lda.py \
  --config_yaml lda/config/training/lda_robocasa.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.vision_encoder_path ${vision_encoder_path} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.max_num_embodiments ${max_num_embodiments} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.obs_horizon ${obs_horizon} \
  --framework.action_model.future_obs_index ${future_obs_index} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${positional_embeddings} \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.training_task_weights ${TRAINING_TASK_WEIGHTS} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 64 \
  --datasets.vla_data.return_vlm_inputs ${return_vlm_inputs} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --trainer.learning_rate.base 4e-5 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project lda \
  --wandb_entity ${wandb_entity} \
  --is_debug False



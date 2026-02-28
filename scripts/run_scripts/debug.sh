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
base_vlm=/mnt/home/liukai/starVLA/playground/pretrained/vlm/Qwen3-VL-4B-Instruct
vision_encoder_path=/mnt/home/liukai/World-Action-Model/pretrained # should be the parent path of vision encoder ckpt

freeze_module_list='qwen_vl_interface,action_model.vision_encoder' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""
DIT_TYPE="DiT-B"
data_root_dir=/mnt/project
data_mix=all_dataset # should be recorded in data_config.py

obs_horizon=2
state_dim=null
action_dim=138
max_num_embodiments=1
num_layers=8
use_delta_action=true
positional_embeddings=null

repeated_diffusion_steps=1
training_task_weights="[1.0,1.0,1.0,1.0]"

future_obs_index=5
run_root_dir=/mnt/project/world_model/checkpoints/lda # replace with your own path
run_id=debug

pretrained_checkpoint=null # set to null if training from scratch
vision_encoder_type='dinov3'

only_policy=false
policy_and_video_gen=false
only_wo_video_gen=false

export WANDB_MODE=disabled
wandb_entity=your/wandb/entity

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

python lda/training/debug.py \
  --config_yaml lda/config/training/LDA_pretrain.yaml \
  --debug \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.vision_encoder_path ${vision_encoder_path} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.max_num_embodiments ${max_num_embodiments} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.obs_horizon ${obs_horizon} \
  --framework.action_model.future_obs_index ${future_obs_index} \
  --framework.action_model.only_policy ${only_policy} \
  --framework.action_model.policy_and_video_gen ${policy_and_video_gen} \
  --framework.action_model.only_wo_video_gen ${only_wo_video_gen} \
  --framework.action_model.diffusion_model_cfg.num_layers ${num_layers} \
  --framework.action_model.vision_encoder_type ${vision_encoder_type} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${positional_embeddings} \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.training_task_weights ${training_task_weights} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 400000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --trainer.learning_rate.base 4e-5 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project lda \
  --wandb_entity ${wandb_entity} \


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

freeze_module_list='' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""
DIT_TYPE="DiT-B"
# freeze_module_list="qwen_vl_interface.model.model.visual,dino_encoder" # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""

llavadata="asv2_conversation_en,asv2_detailed_description_en"
data_root_dir=/path/to/data/directory
data_mix=data_mix_name # should be recorded in data_config.py

obs_horizon=1
state_dim=58 # if set null, will not use state
action_dim=29
max_num_embodiments=32
use_delta_action=false

run_root_dir=/path/to/save/training/results # replace with your own path
run_id=/run/id

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  lda/training/train_starvla.py \
  --config_yaml lda/config/training/lda_robocasa.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.max_num_embodiments ${max_num_embodiments} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_dim ${action_dim} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project lda \
  --wandb_entity ${wandb_entity} \
  --is_debug False



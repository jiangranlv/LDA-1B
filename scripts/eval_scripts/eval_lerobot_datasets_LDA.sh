# ====================================================
# This script is used to evaluate the policy on the lerobot datasets, with the following steps:
# 1. Evaluate the policy on the datasets, calculate the MSE and gripper MSE
# 2. Visualize the trajectories, save the trajectories as GIFs
# 3. TODO: evaluate the policy on target benchmark
# What you need to do:
# 1. specify the run_id and steps to locate the checkpoint path
# 2. specify the datasets to evaluate
# 3. specify the start_traj and end_traj to evaluate the trajectories
# ====================================================
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export WANDB_API_KEY=8c8c73c90734b2e7292c02bb73313673f5167291

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

# model path
run_root_dir=/path/to/ckpt/dir
run_id=/run/id
steps=120000 # training steps
base_dir=${run_root_dir}/${run_id}
model_path=${base_dir}/checkpoints/steps_${steps}_pytorch_model.pt

action_horizon=16
plot_state=false
seed=42

# datasets
agibot_data_root=/path/to/dataset
data_mix=/data/mix/name

is_delta_action=false # trained using delta eef action or not

# eval config
create_trajectory_video=true
video_output_path=${base_dir}/trajectory_video/steps_${steps}
mkdir -p ${video_output_path}

# eval trajs
trajs=2 # total eval trajs
start_traj=0
end_traj=10000000

plot=true
plot_path=${base_dir}/results/steps_${steps}

# 3D trajs
gt_traj_dir=${plot_path}
traj_names=GT,Pred

python lda/eval/eval_policy.py \
  --config_yaml ${base_dir}/config.yaml \
  --save_plot_path ${base_dir}/plots \
  --seed ${seed} \
  --evaluation.model_path ${model_path} \
  --datasets.vla_data.data_root_dir ${agibot_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --evaluation.action_horizon ${action_horizon} \
  --evaluation.plot ${plot} \
  --evaluation.plot_state ${plot_state} \
  --evaluation.save_plot_path ${plot_path} \
  --evaluation.create_trajectory_video ${create_trajectory_video} \
  --evaluation.video_output_path ${video_output_path} \
  --evaluation.original_video_path ${original_video_path} \
  --evaluation.trajs ${trajs} \
  --evaluation.start_traj ${start_traj} \
  --evaluation.end_traj ${end_traj} \
  --is_delta_action ${is_delta_action}




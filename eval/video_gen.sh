
base_path="/mnt/project/world_model/data/RealWorld"
data_name="Galbot_Pick_Vegetable"
ckpt_base_path="/mnt/project/world_model/checkpoints/visualize/galbot"
ckpt_name="LDA_qwenMMDiT_visualize_pick_vegetable"

ckpt_path="${ckpt_base_path}/${ckpt_name}/checkpoints/steps_35000_pytorch_model.pt"
data_path="${base_path}/${data_name}/data/chunk-000/episode_000000.parquet"
video_path="${base_path}/${data_name}/videos/chunk-000/observation.images.front_head_left/episode_000000.mp4"
config_path="${ckpt_base_path}/${ckpt_name}/config.yaml"
save_path="/mnt/home/liukai/code/LDA/eval/results/galbot/pick_vegetable"
tasks_path="${base_path}/${data_name}/meta/tasks.jsonl"
robot_name=galbot
embodiment_id=28
python /mnt/home/liukai/code/LDA/eval/video_gen.py \
    --ckpt_path ${ckpt_path} \
    --data_path ${data_path} \
    --video_path ${video_path} \
    --config_path ${config_path} \
    --save_path ${save_path} \
    --tasks_path ${tasks_path} \
    --robot_name ${robot_name} \
    --embodiment_id ${embodiment_id}
# 在服务器上运行
NUM_GPUS=4
BASE_PORT=7777
CKPT_PATH="/path/to/ckpt/path"

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python lda/deployment/model_server/server_policy.py \
        --ckpt_path "$CKPT_PATH" \
        --port "$PORT" \
        --use_bf16 \
        > "lda/examples/Robocasa_tabletop/delta_eef_lda_server_gpu${GPU_ID}_port${PORT}.log" 2>&1 & 
done
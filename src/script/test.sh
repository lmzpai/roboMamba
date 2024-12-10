device='0,1,2,3,4,5,6,7'
IFS=',' read -ra GPULIST <<< "$device"
CHUNKS=${#GPULIST[@]}

CUDA_VISIBLE_DEVICES=$device MAMBA_TYPE=JLTASTET torchrun --standalone --nnodes 1 --nproc-per-node $CHUNKS test.py \
    --llm_name mamba-2.8b \
    --vision_encoder CLIP224 \
    --save_prefix 'result/robomamba' \
    --checkpoint 'path_to_checkpoint.pth' \
    --run_type VLM \
    --dataset robovqa
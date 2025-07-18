export CUDA_VISIBLE_DEVICES="0,1,2,3" # set gpu(s)

WORLD_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

EXPID=$(date +"%Y%m%d_%H%M%S")
HOST='127.0.0.1'

python train.py \
    --config 'configs/train.yaml' \
    --output_dir 'results' \
    --checkpoint 'ALBEF_4M.pth' \
    --launcher pytorch \
    --rank 0 \
    --log_num ${EXPID} \
    --dist-url tcp://${HOST}:10966 \
    --token_momentum \
    --world_size $WORLD_SIZE \
    --model_save_epoch 5

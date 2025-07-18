export CUDA_VISIBLE_DEVICES="0" # set device
HOST='172.0.0.1'
PORT='13718'
NUM_GPU=1

python test.py \
--checkpoint_dir "/path/to/your/checkpoint_best.pth" \
--config 'configs/test.yaml' \
--output_dir 'results' \


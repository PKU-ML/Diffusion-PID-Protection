### Generate images protected by PID

### SD v2.1
# export MODEL_PATH="stabilityai/stable-diffusion-2-1" 

### SD v1.5
export MODEL_PATH="runwayml/stable-diffusion-v1-5"
export MODEL_PATH="../PID_diffusion/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"

### Data to be protected
export INSTANCE_DIR="./data/clean_data/instance_1"

### Path to save the protected data
export DREAMBOOTH_OUTPUT_DIR="./data/PID_instance"


### Generation command
# --max_train_steps: Optimizaiton steps
# --attack_type: target loss to update, choices=['var', 'mean', 'KL', 'add-log', 'latent_vector', 'add'],
# Please refer to the file content for more usage

CUDA_VISIBLE_DEVICES=0 python PID.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --resolution=512 \
  --max_train_steps=1000 \
  --center_crop \
  --eps 12.75 \
  --attack_type add-log


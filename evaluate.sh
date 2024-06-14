### Path to the clean training data
export CLEAN_DATA="./data/clean_data/instance_1"

### Path to the generated images
export GENERATED_DATA="" 

### Path to save the evaluation logs
export LOG_DIR=$GENERATED_DATA

### Run evaluation
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --clean_image $CLEAN_DATA \
    --generated_image $GENERATED_DATA \
    --log_dir $LOG_DIR \
    --size 512 

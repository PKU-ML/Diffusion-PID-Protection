### Trianing model

### SD v2.1
# export MODEL_PATH="stabilityai/stable-diffusion-2-1" 

### SD v1.5
export MODEL_PATH="runwayml/stable-diffusion-v1-5"

### Path to save the class-regularization images
export CLASS_DIR="./data/class_images"
 
### Training data
export INSTANCE_DIR="./data/clean_data/instance_1"

### Path to save the model
export DREAMBOOTH_OUTPUT_DIR="DREAMBOOTH_LoRA_clean"

### Training command
# --instance_prompt: Prompt used when fine-tuning the model
# --num_class_images: Number of class-regularization images
# --max_train_steps: Training steps
# --rank: Rank for the adapter
# --validation_prompt: Prompt used when generating images 
# --train_text_encdoer: whether to fine-tune the text-encoder
# More usage can be found at https://huggingface.co/docs/diffusers/en/training/lora


CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --rank 32 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=800 \
  --mixed_precision bf16 \
  --num_validation_images 10 \
  --validation_prompt="a photo of sks person" \
  --checkpointing_steps 400 \
  --train_text_encoder



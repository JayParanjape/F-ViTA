export MODEL_ID="timbrooks/instruct-pix2pix"
# export DATASET_ID="jay-jnp/osu"
export DATASET_ID="osu-local"
# export OUTPUT_DIR="checkpoints/tmp"
export OUTPUT_DIR="checkpoints/osu_finetuned_ablation_onlymasks"


accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --use_ema \
  --resolution=256 --random_flip \
  --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=20000 \
  --checkpointing_steps=1000 --checkpoints_total_limit=20 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --seed=42 \
  --original_image_column="visible_image" \
  --edit_prompt_column="edit_instruction" \
  --edited_image_column="thermal_image" \
  --output_dir=$OUTPUT_DIR \
  --use_ibl=False \
  --use_extra_mse=False \
  --use_masks=True \
  --use_boxes=False \
  --use_text=True
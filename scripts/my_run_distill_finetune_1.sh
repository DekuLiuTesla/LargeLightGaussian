#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6526

# Datasets
declare -a run_args=(
    "block_building_all_lr_c20_loss_10_lr2_r4_40_lr64"
    "block_building_all_lr_c20_loss_10_lr2_r4_50_lr64"
    "block_building_all_lr_c20_loss_10_lr2_r4_60_lr64"
    # "kitchen"
    # "room"
    # "stump"
    # "garden"
    #  "train"
    # "truck"
)

# activate psudo view, else using train view for distillation 
declare -a virtue_view_arg=(
  "--augmented_view"
)
# compress_gaussian/output5_prune_final_result/bicycle_v_important_score_oneshot_prune_densify0.67_vpow0.1_try3_decay1
# compress_gaussian/output2
for arg in "${run_args[@]}"; do
  for view in "${virtue_view_arg[@]}"; do
    # Wait for an available GPU
    while true; do
      gpu_id=$(get_available_gpu)
      if [[ -n $gpu_id ]]; then
        echo "GPU $gpu_id is available. Starting distill_train.py with dataset '$arg' and options '$view' on port $port"
        CUDA_VISIBLE_DEVICES=$gpu_id nohup python large_distill_train.py \
          -s "data/mill19/building-pixsfm/train" \
          -m "output/${arg}_distill" \
          --start_checkpoint "output/${arg}/chkpnt30000.pth" \
          --iteration 40000 \
          --eval \
          --teacher_model "output/${arg}/chkpnt30000.pth" \
          --new_max_sh 2 \
          --position_lr_init 0.0000003 \
          --position_lr_final 0.000000003 \
          --scaling_lr 0.000075 \
          --position_lr_max_steps 40000 \
          --enable_covariance \
          $view \
          --port $port > "logs_prune/distill_${arg}_${view}.log" 2>&1 &

        # Increment the port number for the next run
        ((port++))
        # Allow some time for the process to initialize and potentially use GPU memory
        sleep 60
        break
      else
        echo "No GPU available at the moment. Retrying in 1 minute."
        sleep 60
      fi
    done
  done
done
wait
echo "All distill_train.py runs completed."

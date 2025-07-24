#!/bin/bash

# > Default arguments - can be overriden by environment variables:
# architecture to train, must be compatible with the Llama architecture
model=${MODEL:-"/media/dky/Elements/checkpoints/llama/ArmoRM-llama3-8b-v0.1"}   # replace with your actual model path
# total batch size across all devices with gradient accumulation
bsz=${BSZ:-32}
# number of sequences per device
seq=${SEQ:-1}
# peak learning rate
lr=${LR:-5e-5}
# number of epochs
epochs=${EPOCHS:-1}
# warmup ratio
warmup=${WARMUP:-0.1}
# save model every n steps
save_steps=${SAVE:-5000}
# suffix to append to run name
suffix=${SUFFIX:-"multi-turn-dialogue"}
# only predict labels with certain confidence
confidence=${CONFIDENCE:-0.0}
# temperature applied to labels
labeltemp=${LABELTEMP:-2.0}
# sensitivity and specificity learning rate
ss_lr=${SS_LR:-1e-2}
# initial sensitivity and specificity values
initial_sensitivity=${INIT_SENS:-0.5}
initial_specificity=${INIT_SPEC:-0.5}

num_gpus=${NUM_GPUS:-1}

# deepspeed_config_file="../config"

run_name="multi_turn_$(basename $model)_bsz${bsz}_lr${lr}_ss_lr${ss_lr}_sens${initial_sensitivity}_spec${initial_specificity}_epochs${epochs}_warmup${warmup}_conf${confidence}_labeltemp${labeltemp}${suffix}"
out_dir="./results/multi_${run_name}"
mkdir -p $out_dir

header="torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_addr 127.0.0.1 \
    --master_port 9902 \
    Multi_Turn_Train.py "

# export WANDB_PROJECT="multi-turn-dialogue-evaluation"
# export WANDB_DIR=$out_dir
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export DS_SKIP_CUDA_CHECK=1

base_arguments=(
    # --report_to wandb
    --report_to none
    
    --do_train
    --model_name_or_path $model
    
    --initial_sensitivity $initial_sensitivity
    --initial_specificity $initial_specificity
    --ss_learning_rate $ss_lr
    
    --config_name $model
    --config_overrides ""
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --log_level info
    --logging_steps 1
    --disable_tqdm false
    --save_strategy "steps"
    --save_steps $save_steps
    --dataloader_num_workers 2
    --cache_dir .cache
    --overwrite_output_dir
    --remove_unused_columns false
    --use_fast_tokenizer false
    --gradient_checkpointing true

    --num_train_epochs $epochs
    --max_length 8192
    --per_device_train_batch_size $seq
    --gradient_accumulation_steps $(($bsz / $seq / $num_gpus))
    --learning_rate $lr
    --max_grad_norm 1.0
    --weight_decay 0.1
    --warmup_ratio $warmup

    --use_flash_attention_2 false
    --bf16_full_eval
    --bf16
    --ddp_find_unused_parameters false
    --ddp_timeout 36000000

    --label_field "Overall"
    
    --confidence_threshold $confidence
    --label_temperature $labeltemp

    --train_datasets_dir ../data/P^2-MTD  # replace with your actual train data path
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}"

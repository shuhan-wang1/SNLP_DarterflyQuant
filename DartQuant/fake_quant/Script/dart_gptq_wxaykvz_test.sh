#!/bin/bash

# 检查是否提供了8个参数：GPU编号
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL> <W_BITS> <A_BITS> <KV_BITS> <R2_PATH> <R1_PATH>"
    exit 1
fi

# 获取传入的参数
GPU_ID=$1
MODEL=$2
R1_PATH=$7
R2_PATH=$6
W_BITS=$3
A_BITS=$4
KV_BITS=$5

# 提取“/”后面的字符串
MODEL_NAME=${MODEL##*/}

# 执行命令并使用传入的参数
CUDA_VISIBLE_DEVICES=${GPU_ID} python main_for_test.py \
    --model ${MODEL} \
    --fuse_norm \
    --use_r1 \
    --r1_path ${R1_PATH} \
    --use_r2 offline \
    --r2_path ${R2_PATH} \
    --use_r3 \
    --use_r4 \
    --w_groupsize 128 \
    --w_clip \
    --a_asym \
    --a_clip_ratio 0.9 \
    --w_bits ${W_BITS} \
    --a_bits ${A_BITS} \
    --k_bits ${KV_BITS} \
    --v_bits ${KV_BITS} \
    --k_groupsize 128 \
    --v_groupsize 128 \
    --k_asym \
    --v_asym \
    --o_per_head \
    --percdamp 0.1 \
    --no-w_ft \
    --ft_percdamp 0.0 \
    --save_name dart_w${W_BITS}a${A_BITS}kv${KV_BITS}_128nsamples \
    --distribute \
    --ppl_eval \
    --ppl_eval_batch_size 1 \
    --ppl_eval_dataset wikitext2 ptb c4 \
    --lm_eval \
    --lm_eval_batch_size 2 \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai social_iqa openbookqa mmlu \
    # --load_qmodel_path ${LOAD_QMODEL_PATH} \

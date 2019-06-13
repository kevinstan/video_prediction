# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env bash
cd ..
python -u run.py \
    --is_training True \
    --dataset_name action \
    --train_data_paths data/kth \
    --valid_data_paths data/kth \
    --pretrained_model kth_e3d_lstm_pretrain/model.ckpt-200000 \
    --log_dir logs/_kth_finetune_residuals \
    --save_dir checkpoints/_kth_finetune_residuals \
    --gen_frm_dir results/_kth_finetune_residuals \
    --model_name original_e3d_lstm_residuals \
    --allow_gpu_growth True \
    --img_channel 1 \
    --img_width 128 \
    --input_length 10 \
    --total_length 30 \
    --filter_size 5 \
    --num_hidden 64,64,64,64 \
    --patch_size 8 \
    --layer_norm True \
    --reverse_input False \
    --sampling_stop_iter 100000 \
    --sampling_start_value 1.0 \
    --sampling_delta_per_iter 0.00001 \
    --lr 0.00001 \
    --batch_size 2 \
    --max_iterations 200000 \
    --display_interval 500 \
    --test_interval 500 \
    --snapshot_interval 500 \
    --log_interval 500

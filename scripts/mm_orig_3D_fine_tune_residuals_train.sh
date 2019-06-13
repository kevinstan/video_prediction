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
    --dataset_name mnist \
    --train_data_paths data/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths data/moving-mnist-example/moving-mnist-valid.npz \
    --log_dir logs/_mm_orig_3D_residuals_fine_tune_l2only_second \
    --pretrained_model mm_e3d_lstm_pretrain/model.ckpt-80000 \
    --save_dir checkpoints/_mm_orig_3D_residuals_fine_tune_l2only_second \
    --gen_frm_dir results/_mm_orig_3D_residuals_fine_tune_l2only_second \
    --model_name original_e3d_lstm_residuals \
    --allow_gpu_growth True \
    --img_channel 1 \
    --img_width 64 \
    --input_length 10 \
    --total_length 20 \
    --filter_size 5 \
    --num_hidden 64,64,64,64 \
    --patch_size 4 \
    --layer_norm True \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_delta_per_iter 0.00002 \
    --lr 0.000001 \
    --batch_size 4 \
    --max_iterations 30000 \
    --display_interval 500 \
    --test_interval 500 \
    --snapshot_interval 500 \
    --log_interval 500



# --pretrained_model checkpoints/_mnist_e3d_lstm_1700/model.ckpt-1700 \

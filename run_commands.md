# stage-age 已有实验运行命令汇总

本文档记录当前项目中已经实现或已经跑过的主要实验入口。默认在 `us` 虚拟环境运行。

所有命令默认工作目录：

```bash
cd /home/szdx/LNX/stage-age
```

推荐统一写法：

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.<module> <args>'
```

也可以先进入环境：

```bash
cd /home/szdx/LNX/stage-age
conda activate us
export PYTHONPATH=src
```

说明：

- 当前主分组为 `18-44 / 45-64 / 65-100`，即 `bins=[18,45,65,101]`。
- 后续正式对比优先看 multi-seed subject-level 结果，不建议只看单次 split。
- 现有训练脚本会新建带完整时间戳的输出目录，不应覆盖已有 `outputs`。
- 多 GPU 实验不是 DDP，而是一张 GPU 跑一个独立 seed-run。

## 1. ResNet18 单次 baseline

### 45/60 历史分界

对应历史输出已归档到 `outputs/failure/20260511_152105_resnet18_baseline`，不建议继续作为主线。

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.train --config configs/resnet18.json'
```

### 45/65 单次 split

对应已有输出：

- `outputs/20260511_152330_resnet18_age45_65`
- subject macro-F1: `0.6488`
- 备注：这是单次 subject-level split 的较好结果，后续仍以 multi-seed 稳定性为准。

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.train --config configs/resnet18_age45_65.json'
```

## 2. Multi-seed 主线对比

统一入口：`stage_age.run_multiseed_experiments`

默认数据、标签、分组和 split 比例已经写在脚本内：

- image_dir: `/home/szdx/LNX/data/TA/Healthy/Images`
- characteristics: `/home/szdx/LNX/data/TA/characteristics.xlsx`
- sheet_name: `Blad1`
- split: `0.70/0.15/0.15`
- seeds: 通常使用 `42 43 44`

### ResNet18 baseline

对应已有输出：

- `outputs/20260511_160714_resnet18_baseline_age45_65_multiseed`
- subject macro-F1: `0.6313 ± 0.0417`

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_multiseed_experiments --model resnet18_baseline --seeds 42 43 44 --epochs 30'
```

### USFM linear probe

对应已有输出已归档：

- `outputs/failure/20260511_161304_usfm_linear_probe_age45_65_multiseed`
- subject macro-F1: `0.5802 ± 0.0570`
- 备注：线性探针偏弱，后续不作为主线。

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_multiseed_experiments --model usfm_linear_probe --seeds 42 43 44 --epochs 30'
```

### USFM MLP probe

对应已有输出：

- `outputs/20260511_161907_usfm_mlp_probe_age45_65_multiseed`
- subject macro-F1: `0.6434 ± 0.0225`
- 备注：冻结 USFM，只训练 MLP head，接近主线但略低。

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_multiseed_experiments --model usfm_mlp_probe --seeds 42 43 44 --epochs 30'
```

### USFM partial last block

对应已有输出：

- `outputs/20260511_162523_usfm_partial_last_block_age45_65_multiseed`
- subject macro-F1: `0.6471 ± 0.0237`
- subject balanced accuracy: `0.6829 ± 0.0325`
- 备注：当前主线模型，只解冻 `encoder.blocks.11` 和 `encoder.fc_norm`。

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_multiseed_experiments --model usfm_partial_last_block --seeds 42 43 44 --epochs 30'
```

## 3. ResNet18 小规模超参寻优

统一入口：`stage_age.run_resnet18_hparam_search`

### 一次性跑全部 6 组

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_resnet18_hparam_search --configs resnet18_reg_a resnet18_reg_b resnet18_freeze12 resnet18_discriminative_lr resnet18_layer4_only resnet18_layer4_only_strongwd --seeds 42 43 44 --epochs 30'
```

### 只复跑保留价值较高的两组

对应已有输出：

- `outputs/20260511_164847_resnet18_reg_b_age45_65_hparam`
- `outputs/20260511_164847_resnet18_discriminative_lr_age45_65_hparam`
- 二者 subject macro-F1 均约 `0.6259`
- 备注：稳定性有改善，但均值未超过 ResNet18 baseline 和 USFM 主线。

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_resnet18_hparam_search --configs resnet18_reg_b resnet18_discriminative_lr --seeds 42 43 44 --epochs 30'
```

### 单独复跑某个 ResNet18 配置

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_resnet18_hparam_search --configs resnet18_reg_b --seeds 42 43 44 --epochs 30'
```

可选配置名：

- `resnet18_reg_a`
- `resnet18_reg_b`
- `resnet18_freeze12`
- `resnet18_discriminative_lr`
- `resnet18_layer4_only`
- `resnet18_layer4_only_strongwd`

## 4. USFM label smoothing / ordinal 阶段

统一入口：`stage_age.run_next_stage_experiments`

这些实验已经归档到 `outputs/failure`，主要用于记录失败路线，不建议继续主攻。

对应已有输出：

- `outputs/failure/20260512_170016_usfm_partial_ls_lr1e5_age45_65_multiseed`: subject macro-F1 `0.6077 ± 0.0202`
- `outputs/failure/20260512_170016_usfm_partial_ls_lr3e5_age45_65_multiseed`: subject macro-F1 `0.6323 ± 0.0271`
- `outputs/failure/20260512_170016_usfm_partial_ls_lr5e5_age45_65_multiseed`: subject macro-F1 `0.6307 ± 0.0343`
- `outputs/failure/20260512_170016_usfm_partial_ordinal_lr3e5_age45_65_multiseed`: subject macro-F1 `0.5773 ± 0.0290`

### 一次性跑全部 label smoothing + ordinal

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment all --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

### 单独复跑

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment partial_ls_lr1e5 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment partial_ls_lr3e5 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment partial_ls_lr5e5 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment ordinal --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

## 5. USFM 回归再分箱 / focal loss / 中间类加权阶段

统一入口：`stage_age.run_next_stage_experiments`

对应已有输出：

- `outputs/20260512_173950_usfm_partial_regression_binning_age45_65_multiseed`
- `outputs/20260512_173950_usfm_partial_focal_loss_age45_65_multiseed`
- `outputs/20260512_173950_usfm_partial_midclass_weight_1p3_age45_65_multiseed`

结果摘要：

- `usfm_partial_regression_binning`: subject macro-F1 `0.6139 ± 0.0279`，subject MAE/RMSE/Pearson `8.5001 / 10.4698 / 0.7577`，45-64 recall 较高但整体分类下降。
- `usfm_partial_focal_loss`: subject macro-F1 `0.6300 ± 0.0335`，收益不明显。
- `usfm_partial_midclass_weight_1p3`: subject macro-F1 `0.6383 ± 0.0599`，中间类改善但稳定性不足。

### 一次性跑全部三组

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment all_next --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

### 单独复跑 regression binning

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment regression_binning --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

### 单独复跑 focal loss

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment focal_loss --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

### 单独复跑 45-64 中间类权重上调

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment midclass_weight_1p3 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

## 6. 固定 manifest 的 USFM linear probe 历史公平对比

这个入口用于复用 ResNet18 单次 45/65 run 的 manifest，不重新划分数据。主要用于早期和单次 ResNet18 做严格同 split 对比。

固定 manifest：

```text
/home/szdx/LNX/stage-age/outputs/20260511_152330_resnet18_age45_65/manifest.csv
```

复跑命令：

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.train_usfm_linear_probe --epochs 30 --batch-size 32 --num-workers 8 --manifest-path /home/szdx/LNX/stage-age/outputs/20260511_152330_resnet18_age45_65/manifest.csv --checkpoint-path /home/szdx/LNX/stage-age/USFM_latest.pth --adapter-path /home/szdx/LNX/usage_predict_autoresearch/usfm_adapter.py --output-dir outputs/usfm_linear_probe_fixed_manifest_age45_65'
```

## 7. USFM LDL 对比实验

统一入口：`stage_age.run_next_stage_experiments`

LDL 三组实验：

- `ldl_boundary_w3`: 45/65 岁边界附近手写 soft label，远离边界保持 one-hot。
- `ldl_gaussian_sigma3`: 在 18-100 岁年龄轴生成 sigma=3 的高斯分布，再累加到三类。
- `ldl_multitask_w3_lam0p3`: boundary soft label + 年龄回归辅助头，回归 loss 权重为 0.3。

### 一次性跑全部 LDL

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment all_ldl --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

### 单独复跑 boundary LDL

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment ldl_boundary_w3 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

### 单独复跑 Gaussian LDL

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment ldl_gaussian_sigma3 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

### 单独复跑 LDL multitask

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment ldl_multitask_w3_lam0p3 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

## 8. 当前建议优先级

后续如要继续推进，优先顺序建议：

1. 继续以 `usfm_partial_last_block` 作为主线对照。
2. 新方向优先做 subject-level multi-image pooling。
3. 其次考虑 boundary-aware soft label，而不是普通 label smoothing。
4. ensemble 只作为上限估计，不作为主创新。
5. 已归档到 `outputs/failure` 的路线不建议重复跑，除非有新的明确假设。

## 9. USFM Subject-Level Multi-Image Pooling

统一入口：`stage_age.run_next_stage_experiments`

本阶段是真正的 subject-level training：一个 batch item 是一个 subject 的多张图像，不是 image-level training 后再做 post-hoc average。

实验：

- `subject_mean_pool_k3`: 训练时每个 subject 随机采样 3 张图，共享 USFM encoder 后做 mean pooling。
- `subject_attention_pool_k3`: 同样采样 3 张图，但使用轻量 attention pooling 学习每张图的权重。

推荐一次性运行全部 subject pooling 实验：

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment all_subject_pool --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

单独复跑 mean pooling：

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment subject_mean_pool_k3 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

单独复跑 attention pooling：

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python -m stage_age.run_next_stage_experiments --experiment subject_attention_pool_k3 --seeds 42 43 44 --gpus 0 1 2 3 4 5 --parallel_per_gpu 1 --max_parallel 6'
```

## 10. Subject-Level Feature Profile Analysis

入口：`analysis/feature_profile_analysis.py`

默认建议使用当前 multi-seed 主线 `USFM partial last block` 的 subject-level predictions 做分析：

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python analysis/feature_profile_analysis.py --image_dir /home/szdx/LNX/data/TA/Healthy/Images --experiment_dir /home/szdx/LNX/stage-age/outputs/20260511_162523_usfm_partial_last_block_age45_65_multiseed'
```

也可以指定输出目录：

```bash
conda run -n us bash -lc 'cd /home/szdx/LNX/stage-age && PYTHONPATH=src python analysis/feature_profile_analysis.py --image_dir /home/szdx/LNX/data/TA/Healthy/Images --experiment_dir /home/szdx/LNX/stage-age/outputs/20260511_162523_usfm_partial_last_block_age45_65_multiseed --output_dir /home/szdx/LNX/stage-age/outputs/YYYYMMDD_HHMMSS_feature_profile_analysis'
```

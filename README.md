# Face Mask YOLOv8-Light Experiment Project

这个仓库现在以 Face Mask 检测为正式主线，`VOC_subset` 只用于冒烟验证。工程目标是把同一套数据准备、训练、评估和测速流程统一到 3 条模型线：

- `baseline`: 原生 YOLOv8n 检测结构，`nc=2`
- `mobilenetv2`: MobileNetV2 风格轻量 backbone
- `pcg_ghost`: Ghost backbone + Coordinate Attention + 深层通道收缩的近似 Pyramidal 版本

## 目录约定

- `configs/datasets/`
  - `face_mask.yaml`: Face Mask 正式实验数据配置
  - `voc_smoke.yaml`: VOC 冒烟验证配置
- `datasets/`
  - `face_mask/`: 预处理后的正式数据
  - `VOC_subset/`: 现有 VOC 冒烟数据
- `runs/face_mask/<model_key>/`: Face Mask 训练结果
- `runs/voc_smoke/<model_key>/`: VOC 冒烟训练结果
- `results/`
  - `<model_key>_metrics.csv`: Face Mask 正式评估指标
  - `latency_benchmark.csv`: Face Mask 正式测速结果
  - `model_comparison.csv`: Face Mask 最终对比表
  - `figures/*.png`: Face Mask 定性预测图

## 数据主线

Face Mask 数据固定映射为 2 类：

- `0: with_mask`
- `1: without_mask`

`prepare_data.py` 默认使用 Kaggle 数据集 slug `omkargurav/face-mask-dataset`。脚本会：

1. 尝试通过 Kaggle CLI 下载原始数据到 `datasets/raw/face_mask/`
2. 搜索 Pascal VOC XML 标注
3. 把 `mask_weared_incorrect` 合并到 `without_mask`
4. 固定随机种子 `42`
5. 生成 `train/val/test = 80/10/10`
6. 转换成 YOLO 检测格式并写入 `datasets/face_mask/`

如果你已经手动下载了数据，也可以直接执行：

```bash
python prepare_data.py --dataset face_mask --raw-dir /path/to/raw_face_mask
```

VOC 冒烟配置生成命令：

```bash
python prepare_data.py --dataset voc_smoke
```

## 训练

统一训练入口：

```bash
python train.py --dataset face_mask --model baseline
python train.py --dataset face_mask --model mobilenetv2
python train.py --dataset face_mask --model pcg_ghost
```

VOC 冒烟验证：

```bash
python train.py --dataset voc_smoke --model baseline --epochs 1
python train.py --dataset voc_smoke --model mobilenetv2 --epochs 1
python train.py --dataset voc_smoke --model pcg_ghost --epochs 1
```

默认训练参数：

- `imgsz=640`
- `batch=16`
- `epochs=100`
- `optimizer=auto`
- `lr0=0.01`
- `seed=42`

## 评估与测速

评估所有 Face Mask 模型：

```bash
python evaluate.py --dataset face_mask
```

测速所有 Face Mask 模型：

```bash
python benchmark.py --dataset face_mask
```

评估脚本会输出每个模型的：

- `map50`
- `map5095`
- `params_m`
- `gflops`

测速脚本会输出：

- `latency_ms`
- `fps`

`results/model_comparison.csv` 会把 baseline、精度下降百分比和 `<3%` 判定一起汇总。这里的 `delta_map50_vs_baseline` 是相对 baseline 的百分比降幅。

## 生成论文插图

```bash
python infer.py --dataset face_mask
```

默认会把三条模型线的预测图写到：

- `results/figures/baseline_predictions.png`
- `results/figures/mobilenetv2_predictions.png`
- `results/figures/pcg_ghost_predictions.png`

## 统一入口

也可以用统一入口转发命令：

```bash
python main.py prepare-data --dataset face_mask
python main.py train --dataset face_mask --model baseline
python main.py evaluate --dataset face_mask
```

## 说明

- Face Mask 是唯一正式结论依据
- VOC 只做 smoke test，不进入最终论文对比表
- 自定义模型位于 `ultralytics/ultralytics/cfg/models/custom/`
- 自定义模块注册在 vendored `ultralytics` 源码内

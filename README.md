# VOC YOLOv8-Light 项目使用指南

本项目是一个基于 VOC 数据集的轻量化目标检测实验工程。项目围绕 3 个模型进行统一训练、评估、测速和定性分析：

- `baseline`：原生 YOLOv8n 检测结构
- `mobilenetv2`：使用 MobileNetV2 轻量 backbone 的变体
- `pcg_ghost`：使用 Pyramidal Coordinate-GhostNet 的轻量化变体

项目已经整理为 `VOC-only + CUDA-only` 工作流，适合直接在 Windows + NVIDIA GPU + PyCharm 环境中运行。

## 1. 项目在做什么

本项目的目标是：

**在 VOC 数据集上比较 3 种检测模型的精度、速度和模型规模，找出综合表现最好的模型。**

项目围绕 4 类任务展开：

1. 训练 3 个模型
2. 在 VOC 验证集上统一评估
3. 测试 CUDA 推理速度
4. 生成定性预测图和报告图

## 2. 技术路线

```text
VOC_subset 数据集
    ->
prepare_data.py 校验数据并生成 voc.yaml
    ->
train.py 训练 3 个模型
    ->
evaluate.py 计算 mAP / 参数量 / GFLOPs
    ->
benchmark.py 测试延迟和 FPS
    ->
infer.py 生成定性检测结果图
    ->
generate_report_figures.py 生成报告图
    ->
results/voc/ 输出最终对比表和图表
```

## 3. 目录结构

- [`configs/datasets/voc.yaml`](/D:/DA/Sem2/RT/RT-yolov8--main/configs/datasets/voc.yaml)
  VOC 正式数据配置文件
- [`datasets/VOC_subset`](/D:/DA/Sem2/RT/RT-yolov8--main/datasets/VOC_subset)
  VOC 子集数据
- [`train.py`](/D:/DA/Sem2/RT/RT-yolov8--main/train.py)
  单模型训练入口
- [`evaluate.py`](/D:/DA/Sem2/RT/RT-yolov8--main/evaluate.py)
  统一评估入口
- [`benchmark.py`](/D:/DA/Sem2/RT/RT-yolov8--main/benchmark.py)
  推理测速入口
- [`infer.py`](/D:/DA/Sem2/RT/RT-yolov8--main/infer.py)
  定性结果生成入口
- [`run_full_pipeline.py`](/D:/DA/Sem2/RT/RT-yolov8--main/run_full_pipeline.py)
  全流程一键入口
- [`generate_report_figures.py`](/D:/DA/Sem2/RT/RT-yolov8--main/generate_report_figures.py)
  报告图生成脚本
- [`project_config.py`](/D:/DA/Sem2/RT/RT-yolov8--main/project_config.py)
  全局配置、默认参数和结果字段定义
- [`results/voc`](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc)
  最终结果输出目录

## 4. 环境要求

本项目只支持 CUDA，不支持 CPU 或 MPS 回退。

运行前需要满足：

- Windows
- NVIDIA GPU
- PyTorch 能识别 CUDA
- 项目虚拟环境已可用

推荐解释器：

- [`.venv\Scripts\python.exe`](/D:/DA/Sem2/RT/RT-yolov8--main/.venv/Scripts/python.exe)

## 5. 三个模型分别是什么

### baseline

- 配置文件：[yolov8n_voc_baseline.yaml](/D:/DA/Sem2/RT/RT-yolov8--main/ultralytics/ultralytics/cfg/models/custom/yolov8n_voc_baseline.yaml)
- 结构特点：标准 YOLOv8n，`Conv + C2f + SPPF + Detect`
- 作用：作为精度和速度对比的基线模型

### mobilenetv2

- 配置文件：[yolov8n_voc_mobilenetv2.yaml](/D:/DA/Sem2/RT/RT-yolov8--main/ultralytics/ultralytics/cfg/models/custom/yolov8n_voc_mobilenetv2.yaml)
- 自定义模块：[mobilenetv2.py](/D:/DA/Sem2/RT/RT-yolov8--main/ultralytics/ultralytics/nn/modules/mobilenetv2.py)
- 结构特点：把标准 backbone 替换成 MobileNetV2Block，neck/head 保持 YOLOv8 风格

### pcg_ghost

- 配置文件：[yolov8n_voc_pcg_ghost.yaml](/D:/DA/Sem2/RT/RT-yolov8--main/ultralytics/ultralytics/cfg/models/custom/yolov8n_voc_pcg_ghost.yaml)
- 自定义模块：
  - [coordinate_attention.py](/D:/DA/Sem2/RT/RT-yolov8--main/ultralytics/ultralytics/nn/modules/coordinate_attention.py)
  - Ghost 相关模块复用 vendored Ultralytics
- 结构特点：
  - 使用 GhostConv / C3Ghost
  - 深层加入 Coordinate Attention
  - 深层通道约束为 `48 -> 96 -> 80 -> 64`

## 6. 评估指标

最终对比结果保存在：

- [model_comparison.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/model_comparison.csv)

主要指标如下：

| 指标 | 含义 | 趋势 |
|---|---|---|
| `precision` | 预测为正的目标中有多少是真的 | 越高越好 |
| `recall` | 真实目标中有多少被检测出来 | 越高越好 |
| `mAP50` | IoU=0.5 条件下的平均检测精度 | 越高越好 |
| `mAP50-95` | IoU=0.5 到 0.95 的综合平均精度 | 越高越好 |
| `params_m` | 模型参数量，单位百万 | 越低越轻 |
| `gflops` | 理论计算量 | 越低越轻 |
| `latency_ms` | 单张图像推理延迟 | 越低越好 |
| `fps` | 每秒处理帧数 | 越高越好 |
| `delta_map50_vs_baseline` | 相对 baseline 的 mAP50 跌幅百分比 | 越低越好 |
| `meets_lt3pct_drop` | 是否满足 “mAP50 相对 baseline 跌幅 < 3%” | `True` 为达标 |

### 达标标准

项目使用 `mAP50` 作为主判据，跌幅定义为：

```text
(baseline_map50 - current_model_map50) / baseline_map50 * 100%
```

如果结果小于 `3%`，就认为该模型在精度约束上达标。

## 7. 默认训练参数

默认训练参数定义在 [project_config.py](/D:/DA/Sem2/RT/RT-yolov8--main/project_config.py) 中：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `imgsz` | `640` | 输入图像尺寸 |
| `batch` | `24` | 每次迭代的批大小 |
| `epochs` | `100` | 正式训练轮数 |
| `optimizer` | `auto` | 自动选择优化器 |
| `lr0` | `0.01` | 初始学习率 |
| `seed` | `42` | 固定随机种子 |
| `workers` | `2` | 数据加载线程数 |
| `patience` | `20` | 早停容忍轮数 |
| `cache` | `disk` | 图像缓存模式 |
| `amp` | `True` | 开启混合精度训练 |

## 8. 推荐运行方式

### 一键按顺序跑完整项目

```powershell
.\.venv\Scripts\python.exe run_full_pipeline.py
```

会依次执行：

1. `prepare`
2. `smoke_baseline`
3. `smoke_mobilenetv2`
4. `smoke_pcg_ghost`
5. `train_baseline`
6. `train_mobilenetv2`
7. `train_pcg_ghost`
8. `evaluate`
9. `benchmark`
10. `infer`

### 分步单独运行

```powershell
.\.venv\Scripts\python.exe prepare_data.py --dataset voc
.\.venv\Scripts\python.exe train.py --dataset voc --model baseline --device cuda:0
.\.venv\Scripts\python.exe train.py --dataset voc --model mobilenetv2 --device cuda:0
.\.venv\Scripts\python.exe train.py --dataset voc --model pcg_ghost --device cuda:0
.\.venv\Scripts\python.exe evaluate.py --dataset voc --split val --device cuda:0
.\.venv\Scripts\python.exe benchmark.py --dataset voc --device cuda:0
.\.venv\Scripts\python.exe infer.py --dataset voc --device cuda:0
.\.venv\Scripts\python.exe generate_report_figures.py
```

## 9. PyCharm 中如何运行

推荐只建一个 Run Configuration：

- `Script path`：
  [run_full_pipeline.py](/D:/DA/Sem2/RT/RT-yolov8--main/run_full_pipeline.py)
- `Interpreter`：
  [`.venv\Scripts\python.exe`](/D:/DA/Sem2/RT/RT-yolov8--main/.venv/Scripts/python.exe)
- `Working directory`：
  [`D:\DA\Sem2\RT\RT-yolov8--main`](/D:/DA/Sem2/RT/RT-yolov8--main)
- `Parameters`：
  留空即可

## 10. 输出结果在哪里

### 训练输出

- [`runs/voc/baseline`](/D:/DA/Sem2/RT/RT-yolov8--main/runs/voc/baseline)
- [`runs/voc/mobilenetv2`](/D:/DA/Sem2/RT/RT-yolov8--main/runs/voc/mobilenetv2)
- [`runs/voc/pcg_ghost`](/D:/DA/Sem2/RT/RT-yolov8--main/runs/voc/pcg_ghost)

### 汇总结果

- [baseline_metrics.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/baseline_metrics.csv)
- [mobilenetv2_metrics.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/mobilenetv2_metrics.csv)
- [pcg_ghost_metrics.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/pcg_ghost_metrics.csv)
- [latency_benchmark.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/latency_benchmark.csv)
- [model_comparison.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/model_comparison.csv)

### 定性结果图

- [baseline_predictions.png](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/figures/baseline_predictions.png)
- [mobilenetv2_predictions.png](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/figures/mobilenetv2_predictions.png)
- [pcg_ghost_predictions.png](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/figures/pcg_ghost_predictions.png)

### 报告图

- [01_accuracy_speed_scatter.png](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/report_figures/01_accuracy_speed_scatter.png)
- [02_accuracy_runtime_bars.png](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/report_figures/02_accuracy_runtime_bars.png)
- [03_delta_vs_baseline.png](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/report_figures/03_delta_vs_baseline.png)
- [04_qualitative_comparison.png](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/report_figures/04_qualitative_comparison.png)

## 11. 当前最终结果

| 模型 | mAP50 | mAP50-95 | Params(M) | GFLOPs | Latency(ms) | FPS | 相对 baseline 跌幅 | 是否达标 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.5211 | 0.3317 | 3.0095 | 8.1056 | 7.0460 | 141.92 | 0.00% | 是 |
| mobilenetv2 | 0.4508 | 0.2796 | 1.7398 | 6.4333 | 8.1713 | 122.38 | 13.48% | 否 |
| pcg_ghost | 0.3681 | 0.2172 | 1.0289 | 6.5741 | 10.5379 | 94.90 | 29.36% | 否 |

## 12. 最好的模型是哪个

当前最好的模型是：

**`baseline`**

原因：

- `mAP50` 最高
- `mAP50-95` 最高
- 实测延迟最低
- 实测 FPS 最高
- 是唯一满足 “相对 baseline 的 mAP50 跌幅 < 3%” 的模型

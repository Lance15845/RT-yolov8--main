# QUICKSTART

这是一份面向第一次接触本项目的快速上手说明。

如果你只想知道：

- 这个项目在做什么
- 怎么最快跑起来
- 最终结果在哪里看
- 最好的模型是哪一个

那么只看这份文件就够了。

## 1. 项目目标

本项目在 VOC 数据集上比较 3 个目标检测模型的精度、速度和模型规模：

- `baseline`
- `mobilenetv2`
- `pcg_ghost`

目标是找出在当前实验条件下综合表现最好的模型。

## 2. 三个模型是什么

- `baseline`
  原生 YOLOv8n 检测结构，是基线模型
- `mobilenetv2`
  用 MobileNetV2 轻量 backbone 替换标准 backbone
- `pcg_ghost`
  使用 GhostConv、C3Ghost 和 Coordinate Attention 的轻量化模型

## 3. 环境要求

本项目是：

- `VOC-only`
- `CUDA-only`

必须满足：

- Windows
- NVIDIA GPU
- PyTorch 能识别 CUDA
- 项目虚拟环境可用

推荐解释器：

- [`.venv\Scripts\python.exe`](/D:/DA/Sem2/RT/RT-yolov8--main/.venv/Scripts/python.exe)

## 4. 最快的运行方式

直接运行下面这一条，就会按顺序跑完整个项目：

```powershell
.\.venv\Scripts\python.exe run_full_pipeline.py
```

如果你只想生成报告图，不想重新训练：

```powershell
.\.venv\Scripts\python.exe generate_report_figures.py
```

## 5. 在 PyCharm 中怎么跑

推荐只建一个 Run Configuration：

- `Script path`：
  [run_full_pipeline.py](/D:/DA/Sem2/RT/RT-yolov8--main/run_full_pipeline.py)
- `Interpreter`：
  [`.venv\Scripts\python.exe`](/D:/DA/Sem2/RT/RT-yolov8--main/.venv/Scripts/python.exe)
- `Working directory`：
  [`D:\DA\Sem2\RT\RT-yolov8--main`](/D:/DA/Sem2/RT/RT-yolov8--main)
- `Parameters`：
  留空即可

## 6. 结果在哪里看

- 最终对比表：
  [model_comparison.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/model_comparison.csv)
- 延迟测速表：
  [latency_benchmark.csv](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/latency_benchmark.csv)
- 定性结果图：
  [results/voc/figures](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/figures)
- 报告图：
  [results/voc/report_figures](/D:/DA/Sem2/RT/RT-yolov8--main/results/voc/report_figures)

## 7. 怎么判断模型好不好

项目主要看这些指标：

- `mAP50`
- `mAP50-95`
- `latency_ms`
- `fps`
- `params_m`
- `delta_map50_vs_baseline`

其中最关键的达标标准是：

**相对 `baseline` 的 `mAP50` 跌幅是否小于 `3%`**

## 8. 当前最终结果

| 模型 | mAP50 | mAP50-95 | Params(M) | Latency(ms) | FPS | 跌幅 | 是否达标 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.5211 | 0.3317 | 3.0095 | 7.0460 | 141.92 | 0.00% | 是 |
| mobilenetv2 | 0.4508 | 0.2796 | 1.7398 | 8.1713 | 122.38 | 13.48% | 否 |
| pcg_ghost | 0.3681 | 0.2172 | 1.0289 | 10.5379 | 94.90 | 29.36% | 否 |

## 9. 最好的模型是哪个

当前最好的模型是：

**`baseline`**

原因：

- 精度最高
- 速度最快
- 是唯一满足 `<3%` 跌幅标准的模型

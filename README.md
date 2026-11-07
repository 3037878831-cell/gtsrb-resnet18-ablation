# 改进 ResNet-18 的 GTSRB 交通标志识别

代码仓库：https://github.com/3037878831-cell/gtsrb-resnet18-ablation

本项目比较标准 ResNet-18、CBAM、多尺度空洞卷积、GELU、Mish、GroupNorm
以及组合模型。组合模型由验证集消融结果决定为多尺度卷积 + Mish + BatchNorm。
网络针对 32×32 小图像将 ResNet 的 7×7/stride-2 stem 改为
3×3/stride-1，并去掉输入端最大池化。

## 环境

推荐 Python 3.10、PyTorch 2.5.1、CUDA 12.4。Windows/RTX 4060 示例：

```powershell
conda create -n gtsrb python=3.10 -y
conda activate gtsrb
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib pillow tqdm pyyaml python-docx
```

## 训练

推荐先运行 `powershell -ExecutionPolicy Bypass -File .\download_gtsrb.ps1`。脚本从公开的
Udacity镜像下载并校验GTSRB固定划分（34,799/4,410/12,630）；所有实验使用相同划分。
如果固定划分不存在，训练脚本也可以回退到 torchvision 官方数据并按类别分层切分。

```powershell
python train.py --variant baseline --epochs 30 --batch-size 128
python train.py --variant cbam --epochs 30 --batch-size 128
python train.py --variant multiscale --epochs 30 --batch-size 128
python train.py --variant gelu --epochs 30 --batch-size 128
python train.py --variant mish --epochs 30 --batch-size 128
python train.py --variant groupnorm --epochs 30 --batch-size 128
python train.py --variant combined --epochs 30 --batch-size 128
```

每个实验目录自动生成：

- `config.json`：环境、参数、数据量、参数量和MACs
- `history.csv`：逐轮训练/验证指标
- `best.pt`：按验证集准确率选择的最佳权重
- `metrics.json`：最终测试指标、模型大小、耗时和推理延迟
- `training_curves.png` 与 `confusion_matrix.png`

## 冒烟测试

```powershell
python train.py --variant baseline --epochs 1 --batch-size 16 `
  --workers 0 --limit-train-batches 2 --limit-eval-batches 2
```

报告中的数值应直接取自以上JSON/CSV，不应手工估算。

## 单图预测和可解释性

```powershell
python predict.py path\to\sign.png --checkpoint outputs\baseline\best.pt
python gradcam.py path\to\sign.png --checkpoint outputs\baseline\best.pt
python robustness.py --checkpoint outputs\baseline\best.pt
python aggregate_results.py
```

独立测试与鲁棒性实验均表明 `baseline` 是当前推荐部署模型；`combined`
取得最高验证准确率，可用于复现实验中“验证集最优不等于测试集最优”的对照结论。

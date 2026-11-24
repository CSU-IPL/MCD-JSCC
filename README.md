# Semantic Communication — Partial Source Release

这是项目的部分源码版本，提供基础编码器、解码器、通用网络模块及 AWGN、Rayleigh、Rician 信道实现。

当前版本暂不提供知识蒸馏训练、教师模型、多阶段训练流程、预训练权重、实验日志及论文实验结果，不能用于完整复现论文。基础模型结构（包括其中的注意力模块）包含在本次公开范围内。

## 文件

| 路径 | 内容 |
| --- | --- |
| `models/` | 基础编码器、解码器及收发端组件 |
| `modules.py` | 通用网络层与注意力模块 |
| `channel.py` | 信道仿真 |
| `config.py` | 基础模型配置 |
| `datasets.py`、`Get_datasets.py` | 图像数据读取工具 |
| `distortion.py` | 图像失真指标 |
| `logger_utils.py` | 通用日志工具，TensorBoard 为可选依赖 |
| `demo.py` | 使用随机权重的前向计算示例 |

## 安装与运行

在具有 Python 和 PyTorch 支持的环境中安装依赖：

```sh
python -m pip install -r requirements.txt
python demo.py --channel awgn
python demo.py --channel rayleigh
python demo.py --channel rician
```

示例默认使用 CPU 和随机生成的图像，无需数据集或权重。它仅验证编码、信道传输和解码的张量流程；随机权重的输出不代表模型重建效果。`--rate` 是基础适配器的条件输入，该示例不实现完整系统的码率控制或评估流程。使用 `--device cuda:0` 可选择 GPU。

数据读取工具默认从 `data/` 查找数据，也可设置环境变量 `DATASET_ROOT` 或修改 `config.data_root`。目录约定为 `DIV2K/train`、`DIV2K/test`、`Kodak` 等；数据集需自行准备。

Rician 信道的具体约定见 [RICIAN_CHANNEL.md](RICIAN_CHANNEL.md)。

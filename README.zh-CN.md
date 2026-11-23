# FLARE 项目使用说明

完整命令、结果表和配置说明见 [README.md](README.md)，验收范围见 [VERIFICATION.md](VERIFICATION.md)。

本项目提供 DIOR 与 VisDrone2019-DET 的原版频域增强、PDR 和 CABE 模块，以及对应 Baseline。两套实际实验源码分别保存在 `models/dior` 和 `models/visdrone`，统一入口自动选择对应版本。新版频域和质量引导实验不属于支持的复现配置。

使用顺序为：建立 Linux Python 3.11 环境，安装 CUDA 12.1 编译工具链，运行 `bash setup.sh`，下载数据与权重，然后执行 README 中的训练或验证命令。GPU 驱动与 CUDA 原生扩展依赖目标机器，不能将原服务器二进制环境当作通用环境直接复制。

权重来自 https://www.modelscope.cn/datasets/etheral11/diffudetr 。VisDrone 最佳 Baseline 为 11 万步，训练日志最佳 Full 为 10 万步，保留完整 checkpoint 和 EMA。DIOR 最佳权重保留在原消融归档，下载脚本会抽取对应文件并检查 SHA-256。数据图片需要自行向数据集作者获取。

DIOR 总 batch 为 4，两组使用相同多尺度训练。VisDrone batch 为 2，Full 使用多尺度，Baseline 使用固定尺寸。VisDrone 的 `bbox` 与 `coco_bbox` 是不同评价口径，APs/APm/APl 属于后者，结果表不能拼接两种口径的数值。

本次四个配置已通过 CPU 模型构建，VisDrone 最佳权重已通过 EMA 严格加载。当前服务器没有可见 GPU，尚未完成新发布入口的 GPU 推理、训练和 AP 复验。原始训练数据、权重与任务均未改动。

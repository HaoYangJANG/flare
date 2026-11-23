# 六组消融实验最终归档（2026-10-17）

## 内容与权重选择

本包包括 baseline、frequency_only、pdr_only、no_frequency、no_boundary、full 六组正式实验。所有训练已经按用户要求停止，本次只归档，不恢复训练、不运行GPU验证、不删除原件。

**每组仅保留总体 bbox/AP 最高的一轮完整训练 checkpoint。baseline 采用真实最佳80k，取代上一份五组包使用的第二名65k政策。** APs列取自同一个AP最佳轮，不能把另一轮最高APs混到该权重的成绩中。每组APs最高值/对应轮次在Excel中另列，但不额外保留APs最佳轮权重。

| 实验 | AP排名 | 所选步数 | AP | 同轮APs | 完整训练权重 |
|---|---:|---:|---:|---:|---|
| baseline | 1 | 80,000 | 56.754302 | 19.523639 | model_0079999.pth |
| frequency_only | 1 | 75,000 | 56.879732 | 19.460182 | model_0074999.pth |
| pdr_only | 1 | 75,000 | 57.169182 | 20.861677 | model_0074999.pth |
| no_frequency | 1 | 75,000 | 57.272877 | 20.441296 | model_0074999.pth |
| no_boundary | 1 | 80,000 | 57.275972 | 20.651731 | model_0079999.pth |
| full | 1 | 75,000 | 57.473260 | 20.929642 | model_0074999.pth |

本包与 `/data2/hyjiang/ablation_archives/stopped_ablations_20261016.tar.gz` 是两份独立归档。旧包不改动。本包包含全开，且baseline权重改为80k。

## 对照表和各轮记录

- `documentation/消融实验汇总.xlsx`：主表为六组AP最佳轮对照，包括F/PDR/E开关、所选步数、AP/APs、相对baseline最佳差值、APs最高轮及最后运行步数。第二页为全部完整验证记录及来源。
- `documentation/ablation_summary.csv`、`all_rounds.csv`、`tables.json`、`SUMMARY.md`：表格的CSV、JSON、Markdown版本。CSV为UTF-8 BOM，数值保留原精度；Excel显示四位小数，底层不截断原数值。
- `documentation/<variant>_evaluations.csv/.json`：该组每次完整验证的全部标量与分类AP，原始训练event字段也保留。
- `experiments/<variant>/records/<segment>/`：原始metrics.json、逐步updates.jsonl、log.txt、TensorBoard events、effective config、manifest、初始化/续训证明、运行状态、best_AP和checkpoint标记。
- `experiments/<variant>/weights/`：每组一份所选权重，原生model、ema_state、trainer/optimizer/GradScaler等内容不裁剪。
- 原始60000段和100k续训段均保留。全开包括 `/data1/.../experiments/full` 的0–50k历史及 `/data2/.../full_gpu0_resume50k_20261016` 的迁卡续训历史。50k完整验证由迁卡段补齐，不能把两段当两个重复实验。
- 2026-10-13短暂启动后取消的最早full尝试保存在 `provenance/cancelled_initial_full_20261013`，不是本表的正式全开组，不混入AP排名。
- 各组预期均有17次完整验证（5k～85k），共102次；以实际归档汇总核验为准。无90k/100k完成结果。
- `inference/`保留所有现存COCO预测JSON和`instances_predictions.pth`。后者是预测数据，不是多余训练权重。该目录一般被后续轮次覆盖，**不保证对应所选最佳轮**；各轮数值以metrics/CSV为准，不虚构各轮单独预测。
- best_AP和last_checkpoint是历史原件。last_checkpoint通常指向85k，而本包只保留AP最佳轮；请明确指定weights里的文件，不能盲目依赖历史标记恢复。

## 设置与公平比较

F表示频率特征融合/query调制/前景mask监督。PDR表示三步proposal_absolute rectified box flow、分布式边框细化及加权ensemble `[0.82,0.91,1.0]`。E表示候选边界证据、内外对比、边界残差及refined score calibration。

Baseline是原始DIFFUSION_DINO/DINOCriterion和3-step DDIM，无F/PDR/E。频率单开只启用F，无PDR。E依赖PDR，本次没有不带PDR的E-only组。

共用协议：DIOR-HBB COCO格式20类、共同COCO-derived fresh-DIOR初始化、seed42、物理batch4/累积1、AMP+EMA、训练workers4/验证workers2。训练RandomFlip和短边 `[800,1000,1200,1200]`、max1200；固定1200 EMA验证，无多尺度TTA。每5000 attempted循环验证/保存，恒定原始分组学习率；精确参数与损失权重见各段config/manifest。

baseline/频率单开/正式full从DIOR步0以100000全局预算启动；PDR/去频率/去边界先60000再全状态续到100000预算，最后用户主动停止。full在50k后从GPU3迁到GPU0并验证完整训练状态一致。原生checkpoint不保存RNG/dataloader位置，因此不承诺续训后采样或未来轨迹位级相同。只有单seed，没有统计显著性检验。

**主表是“各组最佳轮”比较，不是同一步数比较。** 公平同轮结果看各轮页。例如75k全开与75k baseline才是同轮；不能拿人为选差的baseline替代真实最佳，也不能把AP最佳与APs最佳拼成一轮。

actual completed_steps=checkpoint零基iteration+1；attempted/applied/skipped分别表示循环次数、成功optimizer更新和AMP跳步，不混为一谈。各组实际停止计数在selection.json，未保存的额外循环不在所选checkpoint里。

## 代码、环境与初始化

`code/source/`保留实际冻结DiffuDETR/detrex与detectron2源代码、配置、原编译扩展。`code/operations*`保留原环境安装、公共初始化、各组训练/续训、全开迁卡适配器、测试与pins。`engineering/`与`launch_logs/`保留预检、smoke与正式运行日志；失败预检和用户Ctrl-C证据保持原样，不误称数值故障。

`environment/prefix/`是实际隔离Python/CUDA依赖的二进制环境快照，`environment/bootstrap/`为原micromamba。`documentation/`另含pip freeze/list、conda explicit/list、Python/CUDA编译器、GCC/G++、glibc、GPU驱动/硬件和OS版本。

实际环境为Linux Python3.11、PyTorch2.3.1+cu121、torchvision0.18.1+cu121、CUDA toolkit12.1.1、GCC11、H200 sm90；精确版本以清单为准。保留原prefix的符号链接、shebang和editable路径，**不是Windows安装包或任意路径可直接运行的便携环境**。原前缀 `/data2/hyjiang/diffudetr_transfer_20261013/runtime/env`。换机器/路径需按清单新建环境并重编译editable detectron2/detrex，不能覆盖当前服务器环境。

`initialization/`另含原COCO checkpoint与zero-DIOR共同初始化，供初始化和pins审计使用，不是额外消融训练权重。共同初始化SHA256 `fbb441c67ad7ae3a0a51033e64f6ef49e4e73b13e2ef4a43acaad77c6440abbe`。

不包含SSH私钥、凭据缓存、其他用户任务、系统镜像、GPU驱动安装包、pip/Git/字节码缓存。源码中的旧AGENTS/CURRENT_STATE/启动指令是历史材料，不能据此自动恢复任何实验。所有停止约束仍有效。

## 数据与恢复

数据图片不重复打包。`data/annotations/`保存实际训练/验证COCO标注及原有尺寸修正前备份（备份不参与训练），`provenance/transfer_metadata/`保留迁移文件清单和原始哈希。

原图路径：`/data2/hyjiang/diffudetr_transfer_20261013/datasets/DIOR_HBB_COCO/coco/images/{train,val}`。

- train标注SHA256 `54a22b75c1069e3a8e4170c85a53c23670dc7d6f0b499317fe480e75435be66e`
- val标注SHA256 `1376306f905552040a705d27086d678bfcbbad9d77648ab778dae2ba087a9504`

新包采用Zstandard压缩，Linux GNU tar支持 `--zstd`：

```bash
cd /data2/hyjiang/ablation_archives
sha256sum -c all_ablations_best_20261017.tar.zst.sha256
zstd -t all_ablations_best_20261017.tar.zst
mkdir /data2/hyjiang/all_ablations_restore_review_20261017
tar --zstd -xf all_ablations_best_20261017.tar.zst -C /data2/hyjiang/all_ablations_restore_review_20261017
cd /data2/hyjiang/all_ablations_restore_review_20261017
sha256sum -c SHA256SUMS
```

解包目录必须全新。归档时不会自动执行恢复，不覆盖任何runtime。原代码/操作路径可由MANIFEST的source字段恢复；各组输出对应selection.json的source_segments。所有绝对路径、GPU UUID和输出存在性检查必须在未来授权运行前重新审查。

未来重新验证应使用所选组/段effective config、原固定1200数据注册、明确weights路径及EMA模式。checkpoint的model是常规模型，ema_state是原验证用EMA，两者不要混用。另设新输出，不直接运行历史启动脚本。当前归档仅进行CPU checkpoint和文件完整性检查，**没有重跑一次GPU AP验证**。

## 文件完整性

根目录MANIFEST.json记录每个文件来源/字节数/SHA256及六组权重选择，SHA256SUMS覆盖全部普通文件（清单自身除外）。包外同名前缀VERIFIED.json记录全tar流逐文件校验，`.tar.zst.sha256`记录压缩包整体校验。只有通过完整校验后才把.partial改为正式文件名。所有原权重、结果、日志和旧归档均保留。

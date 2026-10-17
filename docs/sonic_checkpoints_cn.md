# SONIC 版本选择与缓存

`orcs-sonic-checkpoint` 用于定位或获取官方权重，不加载模型、不导入 PyTorch/MJLab、
不触发任务注册，也不修改现有训练配置。请在服务器执行下载和转换；本地只提交代码与配置。
使用前先安装或更新 ORCS 依赖。

支持官方仓库 `nvidia/GEAR-SONIC` 的三个版本：

- `release`（默认）：`sonic_release/last.pt`
- `v1.1`：`sonic_v1_1/last.pt`
- `low-latency`：`low_latency/last.pt`

## 使用已有文件或缓存

```bash
# 使用服务器现有文件，不下载、不转换。
orcs-sonic-checkpoint --version v1.1 --ckpt /实际路径/sonic_v1_1/last.pt

# 只查服务器的 Hugging Face 缓存，找不到就报错。
orcs-sonic-checkpoint --version v1.1 --local-files-only

# 指定缓存目录和源提交，缺失时允许下载。
orcs-sonic-checkpoint --version low-latency \
  --cache-dir /data/huggingface/hub --revision <commit-sha>
```

`--ckpt` 优先级最高。显式路径不存在、文件为空或仍是 Git LFS 指针时直接报错，
不会回退下载其他版本。本地文件的版本标记为“用户声明，未经内容验证”；
此时 `--revision` 等 Hub 参数不会验证该文件，也不会触发下载。

未指定 `--ckpt` 时，入口使用 `hf_hub_download` 获取所选文件。
未指定缓存目录和 revision 时沿用 SDK 默认行为及缓存环境变量。
普通模式即使命中缓存也可能联网检查版本；需要禁止联网时使用 `--local-files-only`。
直接使用缓存文件，不再复制一份大型权重。

每次成功定位都会输出真实文件位置，并在下面的目录保存小型 JSON 来源记录：

```text
$ORCS_DEPS_ROOT/checkpoints/sonic/<version>/sources/
```

记录包含所选版本、实际路径、文件大小、SHA-256，以及实际 Hub snapshot 提交号；
本地文件则明确记录为用户提供。记录文件按内容命名，重复定位不会覆盖不同的来源记录。
即使处于离线模式，也需要具备写入来源记录目录的权限。
“已找到权重”不代表模型版本语义或训练兼容性已验证。

## 可选：转换原始 release

```bash
orcs-sonic-checkpoint --version release --convert --smpl

orcs-sonic-checkpoint --version release --ckpt /data/original/last.pt \
  --convert --smpl --out-dir /data/experiments/sonic-release-port
```

入口定位已安装 `mocke` checkout 中的 `scripts/port_sonic_checkpoint.py`，
显式传入已有输入文件和输出路径，保留原转换器的检查。
找不到脚本时会提示准备依赖，不会自动修改 `mocke`。转换子进程启用 Hugging Face 离线模式，
避免原转换器的旧下载回退逻辑意外联网。

默认输出：

```text
$ORCS_DEPS_ROOT/checkpoints/sonic/release/ported/
├── last_ported.pt
├── smpl_ported.pt       # 仅 --smpl 时生成
└── conversion.json
```

已有权重、转换记录或锁文件时拒绝覆盖，请为新实验选择新输出目录。
转换先写临时目录，失败不会发布半成品；成功后记录转换脚本、输入和输出的校验值。
进程异常中断可能留下 `.convert.lock`，人工恢复前应确认没有转换进程仍在运行。

`--smpl` 和 `--out-dir` 必须与 `--convert` 一起使用。
本次仅支持 release 的现有转换流程；v1.1 和 low-latency 的转换请求会在下载和写出之前报错：
“当前转换器尚未验证兼容”。没有强制绕过选项，不下载 planner 或 ONNX 模型。

源码目录下也可使用：

```bash
bash scripts/setup/sonic_checkpoint.sh --version v1.1 --local-files-only
```

本功能不修改 `core/rl.py`，现有任务不会自动切换到这些输出。
运行时模型选择、三个版本的训练兼容性仍需单独实现和验证。
本地测试使用模拟缓存和小型占位文件，不代表真实 checkpoint 转换或实机行为已经验证。

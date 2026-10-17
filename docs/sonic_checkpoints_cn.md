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

## 三版本楼梯适配

转换现由 ORCS 的 `core/sonic/port.py` 完成，无需修改 mocke 或 rsl_rl。
转换默认固定官方快照 `6733128a3d8a523b1418b06bca3cdf61c8b0987f`，并核对三个官方
`last.pt` 的 SHA-256。仅结构相同或文件名相同不算匹配；其他版本/自定义权重会明确拒绝。

```bash
orcs-sonic-checkpoint --version release --convert --smpl
orcs-sonic-checkpoint --version v1.1 --convert --smpl
orcs-sonic-checkpoint --version low-latency --convert --smpl
```

输出按版本隔离在 `$ORCS_DEPS_ROOT/checkpoints/sonic/<version>/ported/`。
除权重和来源记录外，`parity.json` 记录转换前后的 latent/token/action 排列数值检查。
转换会根据实际层结构识别输出层；未知控制参数、错误层形状和未处理的归一化参数均报错。
单纯转换成功不代表物理仿真或实机效果已经验证。

仅楼梯任务增加版本选择，其他任务保持原路径。基础 encoder/decoder 权重冻结，
适配器只加在 decoder 上。v1.1 使用机器人 yaw 归一化朝向；low-latency 使用
10 帧、0.02 秒间隔的 G1 参考和 4 帧、0.02 秒间隔的 SMPL 参考。
所有版本仍使用 10 帧本体历史和 50 Hz 控制。

新增任务名：

- `Orcs-PerLoco-GrailStairs-AdaptSonic-V11`
- `Orcs-PerLoco-GrailStairs-AdaptSonic-V11-Smpl`
- `Orcs-PerLoco-GrailStairs-AdaptSonic-LowLatency`
- `Orcs-PerLoco-GrailStairs-AdaptSonic-LowLatency-Smpl`

现有两个楼梯任务名仍表示 release。若没有新的 release 转换权重，允许使用原有 mocke
release 权重。旧 release 训练 checkpoint 没有版本标签时，仅在所有冻结参数逐位一致时允许恢复；
新 checkpoint 保存模型约定指纹，跨版本、跨编码模式或不同基础权重的恢复会被拒绝。

```bash
orcs-pseudo-retarget --scene perloco-grail-stairs --sonic-version v1.1 --all
orcs-view-seeds --scene perloco-grail-stairs --sonic-version v1.1 --source /实际样本目录
play Orcs-PerLoco-GrailStairs-AdaptSonic-V11-Smpl --agent initial
train Orcs-PerLoco-GrailStairs-AdaptSonic-V11-Smpl --env.scene.num-envs 64
```

release seed 仍放在样本根目录 `seed_state.npz`；另外两个版本放在
`seeds/v1.1/seed_state.npz` 和 `seeds/low-latency/seed_state.npz`。
对应来源记录包含基础权重校验值、版本和观测约定。升级前没有模型来源记录的楼梯 seed
需要重新生成；不会自动覆盖其他版本的 seed。

### 验证工具和限制

`tests/verify_official_sonic_onnx.py` 可在服务器对现有官方 v1.1/low-latency ONNX 做 CPU 数值对照。
它不产生训练权重，也不能替代完整 last.pt 的校验。
已发现该固定快照的 low-latency ONNX 实际输入把单帧朝向放在多帧朝向之前，
与发布 observation_config.yaml 的列出顺序不同。验证器按图内实际排列进行比较并报告此差异；
ORCS 直接构建每个 encoder 分支的输入，不使用该全模式 ONNX 打包方式，且未修改官方部署文件。

完整训练验收仍须有有效的已整理楼梯数据、对应 seed 和完整 MJLab/assets 运行环境。
数值测试通过不等于已通过楼梯 reset/step、PPO 收敛或实机测试。

转换先写临时目录，通过检查后才发布；已有输出不覆盖。`--smpl` 和 `--out-dir`
必须与 `--convert` 一起使用。显式 `--ckpt` 不存在时仍然直接报错，不回退下载。
源码目录下的入口仍为 `bash scripts/setup/sonic_checkpoint.sh`。

实际数值与未完成项见[三版本验证记录](validation/sonic_variants_2026-10-17.md)。

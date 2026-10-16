# 感知地形的运动控制

PerLoco 将每段特权参考动作与其对应的地形块配对。AdaptSonic 策略接收地形高度扫描；冻结的 SONIC 基础模型和动作命令机制仍与其他 ORCS 任务共用。

## 前置条件

在已激活的 Python 3.11 环境中，按照根目录的[安装说明](../readme.md#setup)完成安装，并启用 `perloco,smpl` 可选依赖。此外，还需要安装 Git LFS，并准备足够的磁盘空间来存储所选源数据。

准备 GRAIL 的 SMPL 数据需要获得许可的 **SMPL-X v1.1 NPZ** 模型。请在 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de)注册，下载 `SMPL-X v1.1 (NPZ+PKL)`，准备好其中的 `SMPLX_NEUTRAL.npz`。

安装脚本会打印准确的目标路径，并暂停等待该文件就位；如果设置了 `SMPLX_USER` / `SMPLX_PASS` 登录信息，脚本也可以自动下载。

请将男性和女性版本的 `.npz` 文件放在同一目录下：UOLM 重建数据的准备流程优先使用 SMPL-H（由 `ORCS_SMPLH_DIR` 指定，文件为 `smplh/SMPLH_*.pkl`），不可用时回退到 `SMPLX_MALE.npz`。

SMPL-X 默认根目录为 `$ORCS_DEPS_ROOT/body_models`，可以通过 `ORCS_SMPLX_DIR` 覆盖。不需要下载 GRAIL 的代码仓库。

## 下载并整理 GRAIL 数据

在 ORCS 仓库根目录执行：

```bash
bash scripts/setup/perceptive_locomotion.sh --sources grail
```

该命令可以重复执行，并支持中断后继续。它会：

1. 仅下载项目自带清单中指定的 GRAIL 路缘动作系列。
2. 如果设置了 `SMPLX_USER` / `SMPLX_PASS`，下载 `SMPLX_NEUTRAL.npz`；否则等待用户将文件放到打印出的路径中。此步骤仅用于 SMPL。
3. 将地形、机器人动作和 SMPL 动作整理到 `$ORCS_DATA_ROOT/terrain_motions/grail`。

当前清单包含 **8 个路缘地形块上的 63 段动作**。源数据选择配置位于：

```text
src/orcs/tasks/perloco/rosters/grail.toml
```

同一份清单统一控制下载、数据整理、重定向和运行时采样。

如果要下载 PerLoco 支持的两个数据集，执行：

```bash
bash scripts/setup/perceptive_locomotion.sh
```

如果不需要 `-Smpl` 任务，可以添加 `--no-smpl`。这样也会跳过 GRAIL 的 `recon/` 重建数据，以及需要许可的人体模型检查。

查看仅下载、仅整理和非交互运行等选项：

```bash
bash scripts/setup/perceptive_locomotion.sh --help
```

## 验证已准备好的任务

```bash
python -c "import mjlab, orcs; print(orcs.tasks.perloco.SKIP_REASON)"
play Orcs-PerLoco-Grail-AdaptSonic --agent initial --viewer native
```

`SKIP_REASON == {}` 表示没有任务因缺少所需数据而被跳过，相关任务已成功注册。

在执行运动学重定向之前，`-Smpl` 任务出现跳过原因是正常的，因为此时还没有生成 `seed_state.npz`。

可以使用浏览器查看器检查地形与动作是否对齐：

```bash
orcs-view-terrain --source grail
# 打开 http://localhost:8080
```

## 训练

```bash
train Orcs-PerLoco-Grail-AdaptSonic --env.scene.num-envs 4096
train Orcs-PerLoco-OmRe-AdaptSonic --env.scene.num-envs 4096
```

如果需要进行 SMPL 关键点跟踪，并使用 seed 状态进行参考状态初始化（RSI），请继续阅读 [SMPL 重定向说明](smpl_retargeting.md)。

## 故障排查

- **出现 `No module named pxr` 或 `smplx`**：重新运行 `scripts/setup/sync_deps.sh`，不要添加 `--no-smpl`，以确保相关工具和固定版本的 `rsl_rl` 分支在最后的安装步骤中正确安装。
- **Git LFS 指针文件被当成实际数据解析**：安装 Git LFS，然后重新运行安装脚本。不要在完整的 GRAIL 数据集仓库中执行不限定下载范围的 `git lfs pull`。
- **找不到某个任务**：查看 `orcs.tasks.perloco.SKIP_REASON`。注册流程会跳过缺少数据集的任务，而不会让 `import orcs` 失败。

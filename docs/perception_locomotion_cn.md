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

## GRAIL 楼梯：在服务器准备数据

楼梯使用独立清单、输出目录和任务，原 curb 流程保持不变。
`stair1`、`stair2` 分别是官方 `stair_p1`、`stair_p2` 的别名；一次调用不要混合 curb 和楼梯。

激活已安装 `perloco,smpl` 可选依赖的项目环境后，在服务器运行：

```bash
export ORCS_DATA_ROOT=/data/orcs/datasets
export ORCS_DEPS_ROOT=/data/orcs/dependencies
bash scripts/setup/perceptive_locomotion.sh \
  --sources grail --grail-categories stair1,stair2 --yes
```

默认清单 `rosters/grail_stairs.toml` 按名称排序，从每包选择前两个配套文件完整的 family，
每个选择两段动作。“完整”只表示源文件齐全，不代表动作质量合格，也不代表全是上楼动作。
训练前必须回放筛选；`--with-video` 仅下载所选样本的视频。

首次运行会保存 `$ORCS_DATA_ROOT/GRAIL-stairs-selection.json`，记录固定的源提交、
样本 ID 和资产依赖。后续运行复用该清单，不自动跟随远端变化。
`--roster /path/selector.toml` 可指定自定义选择器：使用 `[[samples]]` 列出 category 和 stem，
或修改每包 family 数和每个 family 的片段数。只有明确需要全量数据时才设置 `all_samples = true`。
修改选择器后使用新的 `--selection /path/receipt.json`；也可在选择器中指定 `revision` 提交 SHA。

原始文件存入 `GRAIL-stairs/`，转换结果存入 `terrain_motions/grail_stairs/`。
每条动作对应独立地形 ID，避免不同 take 的缩放、位置不一致却共用地形。
输出目录的 `roster.toml` 只选择本次清单，`stage_report.json` 记录完成、缓存命中和失败样本。
转换失败会阻止楼梯任务注册，而不是静默使用部分数据。

- `--no-smpl`：不下载人体重建，不检查授权人体模型。
- `--skip-clone`：使用已有清单及完整本地文件转换。
- `--skip-stage`：仅下载，不代表新任务已经可运行。
- 下载脚本不会自动生成 seed。

转换器处理 USD 单位、坐标轴、层级变换及物体静态姿态和缩放。
规则实心长方体保持为长方体；其他封闭网格通过 CoACD 转为多个凸碰撞体，不使用整个楼梯的单一凸包。
动态资产、无法确定姿态格式、未支持的实例和无效网格会明确报错。
凸分解仍有近似误差，服务器验收必须检查台阶边缘和碰撞面。
人体与机器人保留现有 GRAIL 按动作阶段对齐的规则；端点检查只能发现明显错位，不能替代回放。

```bash
orcs-view-terrain --source grail_stairs
play Orcs-PerLoco-GrailStairs-AdaptSonic --agent initial --viewer native
orcs-pseudo-retarget --scene perloco-grail-stairs --all
# 将 <tile> 替换为 stage_report.json 中的地形 ID。
orcs-view-seeds --scene perloco-grail-stairs \
  --source /data/orcs/datasets/terrain_motions/grail_stairs/<tile>/level_0.00/sample0
play Orcs-PerLoco-GrailStairs-AdaptSonic-Smpl --agent initial --viewer native
train Orcs-PerLoco-GrailStairs-AdaptSonic-Smpl --env.scene.num-envs 64
```

缺少有效 seed 时，SMPL 任务跳过是正常现象。seed 保存源内容校验记录；动作、元数据或地形改变后，
旧 seed 不再有效。转换结果更新时不会保留上一版本的 seed。
这两个任务仍使用现有 ORCS SONIC 权重和高度扫描，未引入 SONIC v1.1 或深度视觉落脚控制器。

服务器应先验证两个分包的小样本：对齐、碰撞面、初始化、reset/step 和 seed 回放。
一般网格和较大地形网格的显存成本不同，不要直接从 4096 环境起步。
本地契约测试不代表真实数据转换、GPU 运行或策略收敛已经通过验证。

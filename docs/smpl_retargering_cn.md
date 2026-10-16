# SMPL 重定向

PerLoco-GRAIL 和 UOLM 重建动作的处理流程分为两个阶段：

```text
SMPL 动作 → 运动学重定向 → seed_state.npz
SMPL 动作 + seed_state.npz → AdaptSonic 训练 → 动力学重定向
```

seed 仅用于重置时的状态初始化和动作历史。训练参考始终来自源数据：奖励跟踪 SMPL 与 G1 之间的 14 个对应身体点及其速度；对于 UOLM，还包括源数据中的物体轨迹。运动学阶段生成的机器人和物体轨迹不会作为跟踪教师。

所有保存的位置均采用环境局部坐标系。批量处理数据时，每个仿真环境可能位于不同的布局原点，但写入数据时会移除这一原点偏移，并由目标训练场景中的参考状态初始化（RSI）过程加回一次。

## 前置条件

首先完成 [GRAIL 数据下载与整理](perceptive_locomotion.md#fetch-and-stage-grail)。

每个选中的样本都必须包含 `motion.npz` 和 `smpl_motion.npz`。

运动学重定向使用冻结的 SONIC checkpoint，实际运行通常还需要 NVIDIA GPU。

## 对完整 GRAIL 清单执行重定向

```bash
orcs-pseudo-retarget --scene perloco-grail --all
```

该命令将每段待处理动作分配给一个 mjlab 仿真环境，在同一个 GPU 批次中运行整个动作集，只构建一次环境、加载一次 SONIC checkpoint。

不同动作的长度可以不同：较短的动作到达自身末尾后停止记录，较长动作所在的环境继续运行。

命令会在每个 `smpl_motion.npz` 旁写入 `seed_state.npz`，并支持中断后继续。已有 seed 只有同时满足以下条件才会被跳过：

- 数据结构能够正常加载；
- 每一帧都有效；
- 帧数与源数据一致。

只有未通过批量稳定化或有效性检查的环境，才会在独立进程中重试。

如需重新生成全部 seed：

```bash
orcs-pseudo-retarget --scene perloco-grail --all --overwrite
```

调试时可以只处理一段动作：

```bash
orcs-pseudo-retarget --scene perloco-grail \
  --source data/terrain_motions/grail/curb_000/level_0.00/sample0
```

运动学重定向首先将机器人的标称浮动基座对齐到第一帧 SMPL 骨盆姿态，然后进行至少 100 个仿真步的稳定化；如果尚未稳定，会自动继续，直到误差进入规定范围内的低误差稳定平台。

随后，在按质量缩放的虚拟辅助作用下运行冻结的 SONIC。该过程不使用逐帧姿态投影，也不使用已重定向的机器人动作。

如果在全局限制内仍未达到这一稳定平台，就不会写出该动作的结果。

## 检查结果

```bash
orcs-view-seeds --scene perloco-grail \
  --source data/terrain_motions/grail/curb_000/level_0.00/sample0
# 打开 http://localhost:8080
```

查看器会叠加显示原始 SMPL 骨架、仿真中的 G1 运动轨迹、身体点目标、虚拟力标记，以及逐帧跟踪诊断信息。

它读取已保存的结果，不会重新运行策略。

## 基础运行验证与训练

批处理报告 `complete=63/63` 后，分别验证从第零帧初始化和随机 RSI 初始化：

```bash
play Orcs-PerLoco-Grail-AdaptSonic-Smpl --agent initial --viewer native

play Orcs-PerLoco-Grail-AdaptSonic-Smpl --agent initial --viewer native \
  --env.commands.motion.start-from-zero False
```

启动训练：

```bash
train Orcs-PerLoco-Grail-AdaptSonic-Smpl --env.scene.num-envs 4096
```

任务会自动发现具有完整有效 seed 的动作片段。缺失的片段只会从 `-Smpl` 任务中排除，不会改变 `Orcs-PerLoco-Grail-AdaptSonic`。

## UOLM 重建动作

`deps.lock` 固定了位于 `data/reconstructed_motions` 的重建数据集版本。

同步数据后，可以通过一条支持中断后继续的命令，对基础 Chair/Tire 清单执行数据标准化和运动学重定向。该清单仅包含：

- `Data01_Sub01_chair_flip_cam0`
- `Data01_Sub02_tire_roll_cam0`

其他重建的 Chair 和 Tire 动作均被排除。

```bash
bash scripts/setup/sync_data.sh inhouse
orcs-pseudo-retarget --scene uolm --all
```

与 GRAIL 一样，每段待处理动作对应一个 mjlab 环境，在同一次仿真运行中处理。

UOLM 批处理还会为每个环境分配匹配的物体资产，并按照源数据中的第零帧初始化物体状态。基础批次只包含与源数据匹配的 `woodchair2` 和 `tire` 资产，不会创建 Cube 或桌面场景。

只有某个环境未通过稳定化或输出有效性检查时，才会启动子进程。

生成的缓存按行为和场景分组，而不是平铺存放：

```text
data/smpl_motions/uolm/reconstructed/
├── woodchair2-floor/chair_flip/Data01_Sub01_chair_flip_cam0/
├── tire-floor/tire_roll/Data01_Sub02_tire_roll_cam0/
├── small-cube-table/<interaction>/<clip>/
└── big-cube-floor/<interaction>/<clip>/
```

每个末级目录都包含 `smpl_motion.npz`、`object_motion.npz`、`metadata.json` 和 `seed_state.npz`。

Cube 数据集通过独立命名的场景隔离，需要显式处理：

```bash
orcs-pseudo-retarget --scene uolm --motion-set small-cube-table --all
orcs-pseudo-retarget --scene uolm --motion-set big-cube-floor --all
```

源数据适配器会在重定向之前，从重建人体中移除一个通过稳健方法估计、作用于整段动作的竖直方向标定偏差。它不会抹平真实的逐帧运动，也不会移动独立定义的物体。

适配器数据结构发生变化时，支持中断后继续的处理命令会自动使旧版本的已整理样本和 seed 失效。`*-Smpl` 任务会拒绝不完整或过时的选定数据集，而不是静默地使用它们训练。

重定向后，可以检查任意缓存样本：

```bash
orcs-view-seeds --scene uolm --motion-set tire-floor \
  --source data/smpl_motions/uolm/reconstructed/tire-floor/tire_roll/Data01_Sub02_tire_roll_cam0
```

查看器会叠加显示原始 SMPL 和物体动作、仿真生成的 seed 轨迹，以及机器人和物体的虚拟力标记。

物体控制器跟踪源数据中定义的骨盆—物体相对变换，不使用手部弹簧或重力补偿。

训练基础数据集或独立 Cube 场景：

```bash
train Orcs-Uolm-AdaptSonic-Smpl --env.scene.num-envs 4096
train Orcs-Uolm-SmallCubeTable-AdaptSonic-Smpl --env.scene.num-envs 4096
train Orcs-Uolm-BigCubeFloor-AdaptSonic-Smpl --env.scene.num-envs 4096
```

不带场景限定名称的任务选择基础 Chair/Tire 清单。

SmallCubeTable 任务选择现有 `carryflip` 和 `throw` 目录中成功处理的数据的并集；BigCubeFloor 任务仅选择对应的地面数据集。这些特定场景任务不会重复复制或重新生成数据。

每个任务中，各环境的物体物理配置都与所选动作匹配。只有 `small-cube-table` 会创建桌子实体，并将其放在选定动作所定义的目标位置下方。

重建动作没有提供源数据定义的机器人接触标签，因此该任务采用 SMPL 身体点奖励和标准 UOLM 物体姿态/目标奖励，不使用已重定向动作中的接触一致性奖励项。

## 全局辅助增益

与具体任务无关的默认值位于 `src/orcs/core/assisted_retarget.py` 中的 `DEFAULT_ASSISTANCE_GAINS`：

- `response_rate`：`8.0 rad/s`
- `damping_ratio`：`1.0`
- `robot_force_budget_g`：`3.0`
- `object_force_budget_g`：`5.0`

身体点的增益根据对应刚体质量计算，无需为各任务单独调整身体部位或增益。

如果修改了这些默认值，需要使用 `--overwrite` 重新生成 seed。

## 故障排查

- **`orcs-pseudo-retarget: command not found`**：重新激活环境，并运行一次 `uv pip install --no-deps -e .`，刷新命令行入口。
- **`settling did not converge`**：如果存在旧 seed，先使用 `orcs-view-seeds` 检查该样本，然后单独重新运行该样本，以保留完整错误输出。
- **任务未注册**：运行以下命令，并确认对应批次的完成数量：

```bash
python -c "import mjlab, orcs; print(orcs.tasks.perloco.SKIP_REASON); print(orcs.tasks.uolm.SKIP_REASON)"
```

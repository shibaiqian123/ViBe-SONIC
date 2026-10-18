# Release 楼梯 SMPL RGB 初版

外层 ORCS 和内层 ViBe 是两个独立仓库。固定视觉基线提交
`69a406a0249ce1da9ac7eba71a33813302aa1ed6`；两边修改需分别提交。
当前实现增加人体时间轴与视觉任务入口；验收结果另存服务器报告，不以本文作为通过证明。

## 安装与路径（服务器）

```bash
cd /home/cqu023/ViBe-SONIC
bash scripts/setup/install_nested_vibe.sh
export ORCS_DEPS_ROOT="$PWD/dependencies"
export VIBE_DEPS_ROOT="$ORCS_DEPS_ROOT"
export ORCS_DATA_ROOT=/home/cqu023/orcs-data/releases/stairs_smpl_rgb_v1
export VIBE_DATA_ROOT="$ORCS_DATA_ROOT"
export ORCS_SMPLX_DIR="$ORCS_DEPS_ROOT/body_models"
export ORCS_G1_HAND_URDF=/home/cqu023/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf
source .venv/bin/activate
```

不要运行内层原版 sync_deps.sh；它会另外安装官方 ORCS。安装入口检查共享依赖 SHA，
安装视觉依赖后最后重新绑定外层本地 editable 包，并检查安装元数据。

## 独立数据版本

```bash
python -m orcs.cli.stage_terrain_motions --source grail_stairs \
  --root /home/cqu023/orcs-data/GRAIL-stairs \
  --selection /home/cqu023/orcs-data/GRAIL-stairs-selection.json \
  --out "$ORCS_DATA_ROOT/terrain_motions" --smpl --human-timeline --fps 50 --device cuda:0
```

首次执行前确认输出目录没有旧实验；不得复制旧 seed 或接触目标。
人体使用 recon 中 mocap_frame_rate，按源时长重采样（不包含末端重复帧）。
零机器人占位关节只满足 loader/手腕输入接口，非训练目标。
人体与地形共用原坐标，不做平地归零；采用 legacy 体型映射。

先针对关注样本生成 seed（通过 --source 指定其 sample0 目录），检查后再批量：

```bash
orcs-pseudo-retarget --scene perloco-grail-stairs --sonic-version release \
  --source "$ORCS_DATA_ROOT/terrain_motions/grail_stairs/stair_p2_6768415cac75882b/level_0.00/sample0"

# 仅在关注样本接触质量通过后，才使用 --all 批量生成。
```

seed 和训练统一零手腕 tokenizer；seed 仅用于 RSI 与动作历史。基础重定向辅助仍存在，
不使用 stairs-contact-v2。数值有效不能替代接触与外力质量检查。

## 视觉与训练

任务 `Vibe-PerLoco-GrailStairs-ImgFeat-Ext-Smpl` 使用冻结 release SMPL 基础模型、
冻结 FP32 Theia-tiny、可训练交叉注意力提取器与 decoder LoRA。critic 保留高度扫描。
相机为官方 112×63 RGB、42.5° 垂直视角；尚未完成 D435i 实机标定。

先验证 4 环境 reset/step、64 环境短程 PPO 和准确 checkpoint 恢复。
随后单独运行 4096 环境 2 次迭代预检查；失败时不要擅自降低环境数或放宽质量门槛。
全部验收通过才执行正式训练：

```bash
train Vibe-PerLoco-GrailStairs-ImgFeat-Ext-Smpl \
  --env.scene.num-envs 4096 --agent.max-iterations 10000 \
  --agent.logger tensorboard --agent.save-interval 100 \
  --agent.run-name stairs_smpl_rgb_release \
  --log-root "$ORCS_DATA_ROOT/long_runs"
```

正式训练须使用 tmux，重新从基础权重开始，不续接预检查结果。
关闭旧 6006 TensorBoard 前确认 PID 及命令，只关闭可视化服务。
新 TensorBoard 指向实际运行目录、监听 127.0.0.1:6006。
本地转发：`ssh -N -L 127.0.0.1:6006:127.0.0.1:6006 cqu023@10.242.3.229`。
8 段均为训练集，须分别报告 p1/p2；不宣称泛化、实机或平地 VR 控制上楼已经通过。

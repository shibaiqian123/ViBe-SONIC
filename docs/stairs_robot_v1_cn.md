# Release 楼梯机器人参考数据集 v1

服务器数据根：`/home/cqu023/orcs-data/releases/stairs_robot_v1`。
源数据 revision：`40e795761302e611c1e7e3a6caefdd010d56c199`。
制作代码：`2c055462b9678f56377eb846b18f8b78463f1a26`。
固定样本见 `stairs_robot_v1_selection.json`。全部 8 段用于流程验证，没有独立测试集；不能报告陌生地形泛化成绩。

## 环境

每个服务器终端都需要设置：

```bash
cd /home/cqu023/ViBe-SONIC
source .venv/bin/activate
export ORCS_DATA_ROOT=/home/cqu023/orcs-data/releases/stairs_robot_v1
export ORCS_DEPS_ROOT=/home/cqu023/ViBe-SONIC/dependencies
export ORCS_G1_HAND_URDF=/home/cqu023/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf
```

## 已执行结果

- 8/8 转换成功，每段 499 帧、50 Hz、29 关节；浮点数值有限。
- 本版没有 SMPL 文件、seed、seed_history 或接触目标文件；旧目录未覆盖。
- release 机器人参考任务注册成功。
- 8 环境执行 500 步，动作、观测、奖励有限，记录 54 次终止；这不是成功爬楼指标。
- 64 环境 PPO 完成 20 次迭代，使用本地 TensorBoard 日志。
- 24 个 encoder/decoder 基础张量与原始转换权重完全相同；第 0 到第 19 次迭代间，14 个适配器张量均更新。
- 从明确的 model_19.pt 恢复两次迭代成功。当前 CLI 必须使用 `--agent.resume True`，不能省略 True。
- 八个样本查看器加载/几何抽查；没有完成全帧足底间隙、滑移、穿透验收。未启动 256 环境/300 次正式实验。

## 回放

停止占用服务器 8080 的旧查看器后：

```bash
orcs-view-terrain --source grail_stairs \
  --root "$ORCS_DATA_ROOT/terrain_motions/grail_stairs" --port 8080
```

本机转发服务器 8080 后访问 `http://127.0.0.1:8080/`。保持相同 Dex1 环境变量；不要将默认平手资产与本版混用。

## 短程训练与恢复

本次运行目录：
`runs/orcs_perloco_grail_stairs/2026-10-18_13-30-30_stairs_robot_v1_smoke`，相对上述数据根。
最终 checkpoint 为 `model_19.pt`。

```bash
train Orcs-PerLoco-GrailStairs-AdaptSonic \
  --env.scene.num-envs 64 --agent.max-iterations 20 \
  --agent.logger tensorboard --agent.save-interval 5 \
  --agent.run-name stairs_robot_v1_smoke \
  --log-root "$ORCS_DATA_ROOT/runs"
```

明确恢复本次 checkpoint：

```bash
train Orcs-PerLoco-GrailStairs-AdaptSonic \
  --env.scene.num-envs 64 --agent.max-iterations 2 \
  --agent.resume True \
  --agent.load-run 2026-10-18_13-30-30_stairs_robot_v1_smoke \
  --agent.load-checkpoint model_19.pt \
  --agent.logger tensorboard --agent.run-name stairs_robot_v1_resume \
  --log-root "$ORCS_DATA_ROOT/runs"
```

## 证据与未完成项

服务器版本根保存 `build_manifest.json`、`data_checks.json`、`parameter_checks.json`、`smoke_report.json` 及 stage/smoke/train/resume_retry 日志。manifest 记录选择清单、权重和 38 个机器人资产文件的校验值。

p1 每块地形有 369 或 553 个凸碰撞块，p2 为 4–6 个。几何抽查不能替代碰撞精度认证。上楼/下楼策略成功率、全帧接触质量及独立泛化评估仍未验证。在这些检查完成前，不以短程 PPO 跑通宣称模型学会爬楼。

本版数据不要原地重新转换；修改源选择、机器人资产或转换方式时，创建新版本目录。SMPL 训练仍受时间对应关系未验证的限制，不要将旧 seed 复制进本版。

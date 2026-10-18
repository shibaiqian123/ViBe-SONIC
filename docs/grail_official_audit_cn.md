# GRAIL 楼梯官方对照

本入口检查源数据和转换，不执行 SONIC、辅助重定向或 PPO。审计失败时不要重新生成 seed。

## 固定基准

NVIDIA 官方仓库为 https://github.com/NVlabs/GRAIL ，本次固定提交
`da382dcac447261456314c3e3ce15401d5dc1eb0`。程序拒绝不同提交，并按 Git 原始内容或
LFS SHA-256 核验实际使用的官方文件，避免错误 LFS 属性引起的假脏状态。
GMR 的 gitlink 或 vendored tree 标识记录在报告中；不要把 tree SHA 写成上游 GMR commit。
该版本不等于数据发布版本，报告分别记录二者，发布样本所用生成代码尚不能追溯。

## 单样本审计

在服务器安装更新后的 ORCS 后运行，输出目录必须是新的独立目录：

```bash
orcs-audit-grail \
  --source-root /home/cqu023/orcs-data/GRAIL-stairs/data/stair_p2 \
  --sample terrain_stairs__stairs_2065__0001 \
  --official-root /home/cqu023/orcs-grail-official-audit/upstream-sparse \
  --body-models /home/cqu023/ViBe-SONIC/dependencies/body_models \
  --staged-sample /home/cqu023/orcs-data/terrain_motions/grail_stairs/stair_p2_6768415cac75882b/level_0.00/sample0 \
  --out-dir /home/cqu023/orcs-grail-official-audit/focus-new
```

`report.json` 记录来源、检查范围和阻塞项；`replay.npz` 保存服务器上的数值对照。
返回码 2 表示未通过审计，不代表应重试相同配置。当前没有经验证的发布 rollout 与 recon
时间映射，因此审计不会将样本标为 SMPL 训练就绪，也没有强制绕过开关。

人体对照直接执行固定官方文件中的 `forward_smplx` 函数；使用 AST 提取该函数以避免导入
无关的 GMR/Isaac 依赖。分别计算标准人体和 G1 模板分支，保留源帧，不加 warm-up padding、
地面修正或姿态辅助。两分支的差异是模板差异，不能直接认定为 ORCS 转换错误。

## 几何回放

```bash
orcs-audit-grail --view /home/cqu023/orcs-grail-official-audit/focus-new --port 8081
```

通过 VS Code 转发服务器 8081 端口后打开 http://localhost:8081 。人体分支可切换；
人体和机器人帧滑块独立，因为目前不能保证两条时间轴同步。机器人直接播放发布轨迹，
无辅助力，无动力学步进。该查看器地形使用正在审计的 ORCS USD 解释路径，不能用它自身
证明地形转换正确；橙色点显示按固定官方渲染器根变换规则计算的首帧顶点。
完整 Isaac Sim 官方视频及凸分解后的碰撞表面验证尚未完成。

## 对训练的影响

楼梯转换版本提升到 2，原先的首尾归一化明确标为诊断近似。楼梯 SMPL 任务不接受未经验证
的 recon/rollout 对应关系，包括已有 seed；注册报告提示审计原因。机器人参考任务不受此
限制。普通 curb、UOLM 沿用原有行为。正式数据未自动重新转换，旧 seed 未删除。

先完成关注样本的时间来源和人体模板核对，再决定是否迁移正式转换数据及恢复 SMPL 任务。
不要把 `robot.pkl` 中全零的 `smpl_joints` 当作同步人体轨迹。

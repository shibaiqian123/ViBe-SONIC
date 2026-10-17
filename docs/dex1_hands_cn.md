# 使用固定姿态 DEX1 手

默认仍为 ORCS 平板手。设置 `ORCS_G1_HAND_URDF` 后，共用 G1 工厂的查看器、seed 生成和训练改用该 URDF 的 DEX1 手。此设置影响当前进程内所有使用该工厂的任务，不只楼梯任务。

```bash
export ORCS_G1_HAND_URDF=/home/cqu023/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf
orcs-view-terrain --source grail_stairs
```

网格按 URDF 所在目录解析，因此原 `meshes/` 目录需要保留。代码不复制网格，不修改原 URDF，也不依赖本机固定用户名。未设置变量时保持原平板手；路径错误或缺网格时直接报错。

替换范围为两个 wrist-yaw 下的手部外观、碰撞和惯量。保留原有腕部关节、腕部外观、手掌标记、其他身体部件和执行器。原 wrist-yaw 惯量包含旧橡胶手，因此这里使用 URDF 的裸 wrist-yaw 惯量，再加入 DEX1 基座及手指惯量，避免重复计算旧手质量。URDF 中其他腕部及躯干差异不迁移。

4 个 DEX1 滑动关节固定在 URDF 的 q=0，不增加动作维度；这只是固定开合姿态，并不表示完全张开或闭合。手指运动、抓取和独立控制不在此实现范围内。碰撞采用 URDF 指定的碰撞网格，由 MuJoCo 处理为凸碰撞形状，需要回放确认。

生成 seed 和训练时必须在同一终端设置相同的变量。改变手部模型后，应重新生成 seed，并从基础模型开始新实验；不要直接沿用旧手配置的 seed 或训练 checkpoint。目前模型版本检查不自动识别手部资产变化。

```bash
orcs-pseudo-retarget --scene perloco-grail-stairs --sonic-version release --all --overwrite
train Orcs-PerLoco-GrailStairs-AdaptSonic-Smpl --env.scene.num-envs 64
```

以上命令以楼梯数据整理完成、目标任务可注册为前提。手部替换不解决失败的地形样本，也不证明策略已适配新动力学。

恢复默认手：

```bash
unset ORCS_G1_HAND_URDF
```

验证：本地合成模型的编译、关节保持和质量检查通过；服务器真实 URDF/STL 的内存编译通过，nq=36、nv=35，29 个身体关节顺序保持不变。未完成换手后的 GUI 回放、seed、PPO 或实机验证。

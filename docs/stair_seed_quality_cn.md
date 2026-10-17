# 楼梯脚部映射与 seed 接触诊断

本轮新增楼梯专用 `stairs-contact-v2`：接触标注来自 paired robot reference 与实际踏面几何，目标使用机器人脚底到踝部的刚体偏移。10/11 仍是人体脚部关节点，不直接当作鞋底。普通 GRAIL/UOLM 保留 legacy 映射。seed 生成与楼梯 SMPL 训练共享同一映射，不改变 SONIC encoder 输入格式、冻结权重或奖励项/权重。

支撑脚在支撑阶段固定到对应踏面，进入和离开阶段使用 0.1 秒过渡；摆动脚保留人体运动意图。接触目标由 `orcs-prepare-stair-contacts` 生成，缺少有效踏面或配套参考时直接失败，不平移地形或复制参考机器人关节轨迹。

## 更新与生成

更新代码后激活项目环境，设置已有数据、依赖和 DEX1 环境变量；运行 `uv pip install --no-deps -e .` 注册新增 `orcs-seed-quality` 命令。也可使用 `python -m orcs.cli.seed_quality`，无需刷新入口。

本轮不需要重新转换 SMPL 数据。新映射要求重新生成楼梯 seed：

```bash
orcs-pseudo-retarget --scene perloco-grail-stairs --sonic-version release --all --overwrite
```

新来源记录增加映射版本及手部配置内容哈希。旧记录、手部资产变化或不同模型版本的 seed 不参与训练，查看器也会提示重建。正式训练前保持生成 seed 时的手部配置。

替换已有 seed 需要显式 `--overwrite`；成功替换前，旧 seed 和来源记录保存在该样本 `seed_history/<旧 seed 哈希>/`。不要删除这些对照文件。原始数据、转换输出变化仍遵循已有缓存失效机制。

## 接触报告

```bash
# 全部新 seed
orcs-seed-quality --all --out-dir /home/cqu023/orcs-reports/stairs-new

# 配对的原始机器人参考，不需要新 seed
orcs-seed-quality --all --reference --out-dir /home/cqu023/orcs-reports/stairs-reference

# 单样本旧 seed 对照；替换路径中的 tile 和 hash
orcs-seed-quality --source /home/cqu023/orcs-data/terrain_motions/grail_stairs/<tile>/level_0.00/sample0 \
  --seed /home/cqu023/orcs-data/terrain_motions/grail_stairs/<tile>/level_0.00/sample0/seed_history/<hash>/seed_state.npz \
  --out-dir /home/cqu023/orcs-reports/stairs-old
```

工具通过与地形查看器相同的机器人/地形构建器编译单块地形，在 CPU 上做运动学和碰撞查询，不运行策略、不推进动力学。

每个样本输出 JSON 汇总及 NPZ 逐帧数组，包括：

- 脚跟/前脚掌对实际静态碰撞几何的垂直间隙与世界坐标测量点；正值在表面上方，负值在表面下方。它不是任意方向的最近距离，台阶边缘后跟下方可能是低一级踏面。
- 足部倾角、脚部接触穿透，以及几何接触证据。3 mm 是近接触诊断带，不是自动质量门槛；没有测得法向承重力。
- 连续几何接触且测量点未切换时的水平位移速度，作为滑移诊断。摆动、接触切换与首帧记为 NaN，不填成零。
- 双脚无几何接触帧数及最长连续时长、身体点误差、辅助力。原始机器人参考没有跟踪误差和辅助力，这些汇总字段为 null。

源 `foot_contact_probs` 记录形状、范围、原始帧率与整理后帧数/帧率。当前四通道具体含义和时间同步尚未确认，仅做审计，不参与约束或淘汰。端点归一化相位重采样不等于时间戳严格同步。

## 查看器

`orcs-view-seeds` 的楼梯模式增加绿色踝部目标点、粉色脚底测量点，以及脚底间隙、足部倾角、几何接触和穿透读数。原 `valid` 改标为 `finite state`，仅代表数值有限。报告 `quality_pass=null`，不会自动宣布合格。

诊断使用当前机器人手部配置。旧 seed 没有手部来源记录时，不能证明其生成时使用了相同资产；对照报告会保留这种不确定性。

验收结果见 [第一轮服务器验证](validation/stair_foot_mapping_round1.md)。不要把 seed 完成率或更小的身体点误差直接作为训练就绪证明。

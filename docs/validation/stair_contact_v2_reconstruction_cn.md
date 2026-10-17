# 楼梯 `stairs-contact-v2` 重定向

楼梯接触标注只使用配套机器人参考的脚位/脚速与静态踏面几何。候选条件是脚底到踏面的距离不超过 3 mm、踏面法向与竖直方向夹角不超过 30 度，并保留连续帧支撑阶段。源 `foot_contact_probs` 只做诊断，不参与硬约束。

## 准备接触目标

每个 staged sample 需要一份 `robot_reference.npz`，包含：

```text
foot_positions_w       (T,2,3)
foot_velocities_w      (T,2,3)
surface_points_w       (T,S,3)
surface_normals_w      (T,S,3)
ankle_positions_w      (T,2,3)
ankle_to_sole_w        (2,3)
```

生成目标：

```bash
orcs-prepare-stair-contacts \
  --sample /home/cqu023/orcs-data/terrain_motions/grail_stairs/<tile>/level_0.00/sample0 \
  --reference /home/cqu023/orcs-data/terrain_motions/grail_stairs/<tile>/level_0.00/sample0/robot_reference.npz \
  --fps 50 --overwrite
```

命令生成 `stair_contact_targets.npz/json`。它会固定支撑阶段落脚目标，并在进入/离开阶段使用 0.1 秒过渡；无法解析的表面或形状不匹配会直接失败。

## 重新生成与验收

先只处理 `stairs_2065__0001`，确认接触目标报告和 seed，再扩展到全部 8 段：

```bash
orcs-pseudo-retarget --scene perloco-grail-stairs --sonic-version release \
  --source /home/cqu023/orcs-data/terrain_motions/grail_stairs/<tile>/level_0.00/sample0 \
  --overwrite
orcs-seed-quality --source /home/cqu023/orcs-data/terrain_motions/grail_stairs/<tile>/level_0.00/sample0 \
  --out-dir /home/cqu023/orcs-reports/stairs-contact-v2
```

`stairs-contact-v2` seed 会记录支撑通道和脚部映射身份。旧的 `stairs-ankle-v1` seed 不应复用。质量门槛是可信支撑帧中至少 95% 的脚底间隙在 ±10 mm 内、支撑丢失不超过 0.1 秒、穿透不超过 10 mm、稳定支撑滑移速度 P95 不超过 0.1 m/s。达到这些离线门槛也不等于独立平衡或实机安全通过。

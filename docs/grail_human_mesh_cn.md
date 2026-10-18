# GRAIL 人体网格检查

目的：区分骨架点的视觉误解与人体网格确实抬跟。机器人参考路径保持不变；不运行 SONIC、辅助力或 seed 重建，不调整人体高度。

重新运行 `orcs-audit-grail` 原有审计命令，仅将 `--out-dir` 改成新的独立目录。旧 `replay.npz` 没有网格，查看器会明确提示，不能通过重启旧目录补出网格。

本次服务器已生成：`/home/cqu023/orcs-grail-official-audit/focus-mesh-v1`。拉取本次代码后，在服务器运行：

```bash
cd /home/cqu023/ViBe-SONIC
.venv/bin/orcs-audit-grail \
  --view /home/cqu023/orcs-grail-official-audit/focus-mesh-v1 \
  --port 8080
```

先停止旧的 8080 查看器，保留本地到服务器 8080 的 SSH 转发。浏览器访问 `http://127.0.0.1:8080/`。

- `Human branch`：在 `official_standard` 与 `official_g1` 之间切换。标准分支沿用源 scale、以源 trans 为缩放中心；G1 分支沿用官方 G1 模型。没有额外高度或位置补偿。
- `Show human mesh`：显示官方前向输出的真实网格。ORCS 分支只有关节点，不冒充有独立网格。
- `Show ankle / foot labels`：标出 7/8 脚踝、10/11 脚部关节点；它们不是脚跟/脚尖接触点。遮挡脚底时可关闭标签。
- `Play human (independent clock)`：按源人体帧率循环播放完整人体序列，机器人进度保持独立。暂停后可检查具体帧。

两份 `official_*_feet.csv` 分别记录完整 241 帧的左右脚踝高度、脚部关节点高度、踝到脚部轴倾角和脚部关节点速度。`foot_diagnostics.json` 记录范围及限制。这些不是足底接触报告，不据此自动分类“支撑期踮脚”。人体与机器人时序关系仍未验证。

检查时分别观察接近楼梯、抬腿、落脚、支撑、蹬地和楼顶阶段，记录人体源帧号与分支。脚跟在蹬地阶段升高与长期支撑时悬空应区别记录。当前静态楼梯与官方物体全序列仍存在约 1.75 mm 的差异，微小间隙不能过度解释。

验证记录：独立服务器审计生成两套网格，11 项测试通过；浏览器连接成功，人体网格、标签与帧更新显示正常。整段支撑阶段/真实脚底接触尚未完成可靠分类，不能宣称踮脚原因已确认。未改正式数据、旧 seed、SONIC 权重或机器人回放计算。

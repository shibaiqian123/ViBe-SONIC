# SONIC 三版本适配验证记录（2026-10-17）

本记录区分模型/数值验收与物理任务验收。没有运行实机控制或长时间训练。

## 输入来源

- 官方模型仓库：`nvidia/GEAR-SONIC`。
- 固定快照：`6733128a3d8a523b1418b06bca3cdf61c8b0987f`。
- 三份完整 `last.pt` 已在服务器下载，均通过官方 SHA-256 检查。
- release：`e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`。
- v1.1：`af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43`。
- low-latency：`0031ae7db24747445d6eb7c27697640973a837546f0b8763e775143c47d4507c`。
- rsl_rl 验证版本：`215b6f4ceda569d2ca39df64dd679b9af15375cb`，在服务器隔离目录安装，未修改依赖源码。
- PyTorch：`2.7.0+cu128`；数值比较在 CPU FP32 下运行，禁用 TF32。
- 本地未保存真实模型权重。服务器使用临时工作副本验证，不代表正式仓库已经拉取本次修改。

## 完整 checkpoint 转换

对每个版本的 G1、SMPL 分支分别测试 8 组输入，比较原始排列与 MuJoCo 排列。
容差为 `atol=1e-4, rtol=1e-4`，此次没有跨量化边界的样本。

- release：G1 latent 最大误差 `5.960464477539062e-7`，动作误差 `2.86102294921875e-6`；
  SMPL latent 误差 `0`，动作误差 `3.814697265625e-6`。
- v1.1：G1 latent 最大误差 `2.384185791015625e-7`，动作误差 `2.1457672119140625e-6`；
  SMPL latent 误差 `0`，动作误差 `2.1457672119140625e-6`。
- low-latency：G1 latent 最大误差 `2.384185791015625e-7`，动作误差 `2.5033950805664062e-6`；
  SMPL latent 误差 `0`，动作误差 `2.592802047729492e-6`。

三版的编码器隐藏层均为 `[2048,1024,512,512]`。release 动态 decoder 为
`[2048,2048,1024,1024,512,512]`；另外两版为
`[4096,4096,2048,2048,1024,1024,512,512]`。
六份转换权重的运行时元数据/版本约定加载检查通过。

## 实际 LoRA 与恢复验证

六份真实转换权重都通过以下检查：

- 零初始化适配器与冻结基础模型输出误差为 `0`。
- 一次 Adam 优化后，所有冻结参数保持不变，至少一个适配器参数更新。
- 同约定 state_dict 可以恢复；不同约定指纹被拒绝。

这是模型级优化检查，不是一次 PPO 环境训练。
服务器契约与原 seed 格式回归共 11 项通过。
本地相关检查 50 项通过；6 项需 PyTorch 的测试在本地跳过，在服务器执行。

## 官方 ONNX 交叉检查

服务器已有 v1.1 和 low-latency 官方 ONNX。每版、每分支测试 8 组输入：

- v1.1：G1/SMPL token 一致；动作最大误差分别为 `1.6689300537109375e-6`、`1.9073486328125e-6`。
- low-latency：G1/SMPL token 一致；动作最大误差分别为 `2.6226043701171875e-6`、`3.0994415283203125e-6`。

发现该快照 low-latency ONNX 的全模式打包顺序为“单帧朝向在前，多帧朝向在后”，
与随包 observation_config.yaml 的列出顺序不同。验证器使用图中真实顺序并明确记录。
ORCS 直接构造分支输入，不依赖全模式打包；没有修改官方部署文件。

## 尚未完成的物理验收

服务器搜索未发现 `terrain_motions/grail_stairs/` 下的已整理 tile/seed。
因此未运行真实楼梯任务注册加载、环境 reset/step、短程 PPO 或实机验证。
仍需完整 MJLab/mocke/assets 环境、楼梯数据转换及对应版本 seed，然后按使用说明验收。
不能把本记录的数值通过理解为已证明爬楼成功或训练收敛。

## 服务器产物

验证根目录为 `~/orcs-contract-validation/`：

- `dependencies/checkpoints/sonic/<version>/ported/`：六份转换权重、来源记录及 `parity.json`。
- `release_training_contract.json`、`sonic_v1_1_training_contract.json`、`low_latency_training_contract.json`：真实权重优化与恢复报告。
- `onnx_parity_final.json`：官方 ONNX 交叉比较。
- `source/`：临时验证代码副本；正式训练应使用 Git 同步后的代码。

正式训练若要复用上述转换权重，可令 `ORCS_DEPS_ROOT` 指向验证目录的 `dependencies`，
或将完整的版本目录部署到正式 `ORCS_DEPS_ROOT/checkpoints/sonic/` 下；不要只复制某一个文件并混用元数据。

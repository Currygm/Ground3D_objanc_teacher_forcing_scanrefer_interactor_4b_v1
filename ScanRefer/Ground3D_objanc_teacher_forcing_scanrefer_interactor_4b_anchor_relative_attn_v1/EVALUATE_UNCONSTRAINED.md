# ScanRefer Anchor-Conditioned Relative Cross-Attention

新工程：`/home/huangxiang/codes/hx_l/code/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_anchor_relative_attn_v1`。源工程只读。完整设计见docs/PLAN.md。

仅在UniVLG式Decoder第8、9层修改候选Query读取视觉特征的Q和位置logit增量。复用原角色embedding、Q/K/V和xyz_position，不新增损失或评分头。三组为baseline（关闭）、semantic（仅语义）及full（语义＋几何）。

## 训练

只运行一次共享Answer3轮；Qwen LoRA按optimizer更新线性从0升至1e-4。之后三组Joint各20轮，分别新建优化器，Qwen cosine从1e-4降至1e-5。其它组沿用原学习率和阶段cosine；新增标量跟随Decoder LR、WD0。四GPU、batch1、累积2、seed42。初始化完整复用原官方权重/基座/Interactor规则，不加载旧ScanRefer任务权重。

```bash
cd /home/huangxiang/codes/hx_l/code/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_anchor_relative_attn_v1
GPU_IDS=1,2,4,6 bash scripts/train_answer.sh
```

共享Answer预期最终文件（以实际保存为准，阶段轮数独立）：
`outputs/answer_shared/epoch_003_step_00013716.pt`。

```bash
ANSWER="$PWD/outputs/answer_shared/epoch_003_step_00013716.pt"
RUN=baseline GPU_IDS=1,2,4,6 bash scripts/train_joint.sh --answer-checkpoint "$ANSWER"
RUN=semantic GPU_IDS=1,2,4,6 bash scripts/train_joint.sh --answer-checkpoint "$ANSWER"
RUN=full GPU_IDS=1,2,4,6 bash scripts/train_joint.sh --answer-checkpoint "$ANSWER"
```

三条Joint命令由用户分别运行，使用同卡时须等前一条结束；脚本不排队、不自动启动下一组或评测。GPU_IDS也接受逗号分隔UUID，不能与物理编号混用。卡号仅为示例，运行前自行确认资源。所有日志写logs/，缓存写本工程，短TMPDIR为授权根内.arattn_tmp。

每轮保存；Joint20轮预计step91440。`CONFIG=/绝对路径/configs/full.yaml`可替代RUN；两者同时设置时必须一致。

恢复示例（同组、同配置、同四卡world size）：

```bash
RUN=full GPU_IDS=1,2,4,6 bash scripts/train_joint.sh --resume "$PWD/outputs/full/epoch_010_step_00045720.pt"
GPU_IDS=1,2,4,6 bash scripts/train_answer.sh --resume "$PWD/outputs/answer_shared/epoch_001_step_00004572.pt"
```

Joint恢复不要同时传--answer-checkpoint。恢复模型、各rank buffers/RNG、优化器和调度器，拒绝跨组误恢复。已有非空训练产物不能从头覆盖。

## 独立评测

用户自行决定轮次。示例评测Joint第20轮：

```bash
CKPT="$PWD/outputs/full/epoch_020_step_00091440.pt"
RUN=full GPU_IDS=1,2,4,6 bash scripts/evaluate_unconstrained.sh "$CKPT" "$PWD/evaluations/full/epoch020/formal"
```

teacher＋无格式约束generated，全9483条、精确无padding分片、无teacher回退。失败沿用零OBJ隐藏向量解码及strictRaw全分母协议。主看`generated_answer.raw_generated_no_fallback.obj/selected_iou`；Acc25/Acc50为同字典selected_acc字段。unique/multiple子集在`generated_answer.by_subset`（失败记零），teacher有同类字段。子集定义沿用ScanRefer常用NYU18＋others的“原始验证标注场景内已标注实例去重计数”，使用完整原始9508条的类别查表，适配后的正式分母仍9483；没有用适配子集重新定义unique。

保存summary.json、每rank逐样本predictions、原始生成轨迹及rank汇总。候选oracle只作诊断；不允许max-samples替代正式验证。每次输出目录必须为空或不存在。

```bash
RUN=full GPU_IDS=1,2,4,6 bash scripts/evaluate_unconstrained.sh "$CKPT" "$PWD/evaluations/full/epoch020/disable" --anchor-diagnostic disable
RUN=full GPU_IDS=1,2,4,6 bash scripts/evaluate_unconstrained.sh "$CKPT" "$PWD/evaluations/full/epoch020/shuffle" --anchor-diagnostic shuffle_geometry
```

shuffle保留文本、query及语义绑定，场景内置换有效候选几何mask/query对应；每样本固定seed；第8/9层有效候选集合相同时置换一致，集合变化时仅在当层有效候选中重排，避免引入近空mask。没有GT ANC前向通路。

## 验证与限制

CPU验收入口：`bash scripts/check_cpu.sh`，仅CPU和语法检查，不启动任何GPU推理。报告见docs/verification.json。

角色embedding不等于独立实例query；全候选软绑定不保证去重或between关系表达。几何统计的mask停止梯度，绑定权重/query/PE仍可训练。保留原硬mask，因此不能重新访问被硬屏蔽区域。GPU显存、速度、DDP多卡实跑尚未验证；不得把CPU通过等同于训练性能改善。

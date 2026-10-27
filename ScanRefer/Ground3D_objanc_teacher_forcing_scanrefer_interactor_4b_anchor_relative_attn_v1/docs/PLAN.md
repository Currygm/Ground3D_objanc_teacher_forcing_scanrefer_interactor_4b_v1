# ScanRefer：Anchor-Conditioned Relative Cross-Attention 实施计划

更新：2026-10-27（Asia/Shanghai）
状态：规划完成，尚未实施。用户已授权剩余细节由助手判断；下文区分已确认决策与补齐的实现默认值。本次只编写计划，不修改远端代码、不启动GPU任务。

## 1. 目标、工程与实验对照

在Ground3D＋Interactor＋Qwen3-VL-4B＋UniVLG式9层Decoder上，仅改第8、9层候选Query→Visual交叉注意力，让预测ANC的语义和相对空间位置参与目标视觉读取。不新增关系编码器、几何场、mask refinement、offset head或损失。

新工程（已确认）：
`/home/huangxiang/codes/hx_l/code/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_anchor_relative_attn_v1`

只读复制源工程：
`/home/huangxiang/codes/hx_l/code/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_v1`

在224核验hostname=gpu-224、授权根/home/huangxiang/codes/hx_l及所有写入真实路径。仅复制代码、必要配置、脚本及相关测试；不复制输出、缓存、虚拟环境或训练权重。现有数据、模型、Interactor资产和Python环境只读复用；新工程记录其真实路径、版本和hash，不改旧工程。不得修改已运行的关系短语标注任务及其产物。

| RUN | ANC语义融合 | 相对位置增量 | 用途 |
|---|---|---|---|
| baseline | 关 | 关 | 同代码重新训练正式基线 |
| semantic | 开 | 关 | 语义融合消融 |
| full | 开 | 开 | 完整方案 |

三组相同初始化、数据、seed、训练预算及评测协议；不以历史结果代替本次基线。虽然当前只交付三组，两个开关在模型接口上独立，支持后续仅几何配置。首轮只做ScanRefer，不包含Reason3D。

## 2. 原模型边界与数据接口

- 保留Ground3D视觉编码器、32-query Interactor、Qwen LoRA、文本/视觉Adapter、50条共享候选query及原mask/短语匹配头。Decoder hidden256、8头，每头32维。
- Decoder视觉输入仍是LLM前完整superpoint特征，经原视觉Adapter和connector进入Decoder。
- 原层顺序：[Query＋Text]读取Visual → 联合Self-Attention/FFN → Visual读取联合特征/FFN → mask及短语预测。顺序、Value、输出投影及残差主干不变。
- 原角色embedding仍只加在文本特征上，区分QUESTION/OBJ/ANC/OTHER；不再直接重复加到候选query。50条query不是固定OBJ/ANC槽位。
- 训练36569条、验证9483条，原样复用实例级目标、teacher Answer、实体映射和划分，不重新构造文本。关系短语sidecar仅可作诊断，不作模型输入、不等待其完成。
- 为Decoder增加可选的逐样本OBJ/ANC短语token范围和superpoint原始点数。沿用现有token对齐；以实体/标签范围区分多个ANC，禁止把所有ANC角色token合为一个短语。接口仅传范围和计数，不传实体GT ID、GT mask或Hungarian匹配结果。
- teacher模式用teacher文本的实体范围，generated模式用实际生成文本的解析范围；不依赖GT实体名称匹配来决定参照是否参与。现有GT对齐仅用于损失/诊断，不进入选择路径。
- 多实例短语保留全候选分布，不任取一个GT实例，不用GT实例数截断预测。

## 3. 注意力的完整数学定义

以下为一条样本及第l层（l=8或9）的计算。q_j、z_ki、L_kt均取紧邻上一层输出；第8层用第7层，第9层用第8层。j,k=1..50，a=1..A为ANC短语，i为有效superpoint。

### 3.1 候选有效性、中心与实体绑定

x_i是与Decoder视觉特征严格对齐的米制superpoint中心，n_i是该superpoint原始点数。禁止用体素采样后计数替代原始计数。若源缓存不保存计数，先从原点→superpoint映射统计，复用现有映射，不重做分割。

几何mask概率p_ki = stop_gradient(sigmoid(z_ki))。按有效superpoint的原始点数计算：

- m_k = sum_i(n_i p_ki) / sum_i(n_i)。
- m_k ≤ 1e-6的候选视为数值近空；无有效点时不构造中心。
- mu_k = sum_i(n_i p_ki x_i) / sum_i(n_i p_ki)，仅对有效候选计算，分母clamp_min(1e-6)。

所有统计与归一化用FP32，排除padding。NaN/Inf应报错，不能过滤异常掩盖失败。此处没有语义置信度阈值。

对非空ANC短语a的token集合S_a：

- s_ka = mean_{t in S_a}(sigmoid(L_kt))。
- 无效候选分数置0；w_ka = s_ka / sum_r(s_ra)。
- 分数和≤1e-6时该ANC短语无效，否则归一化；不加softmax温度、不硬选TopK。
- q_a = sum_k(w_ka q_k)。

OBJ门控rho_j = mean_{t in S_obj}(sigmoid(L_jt))，不在候选之间归一化。缺失OBJ范围时rho_j=0。s、w、rho、q_a均保留梯度；只有几何统计读取的mask停止梯度。

### 3.2 多ANC权重

使用当前层joint_visual注意力原Q/K内容投影及原偏置，参照评分不加入query槽位位置项：

- score_ja = mean_h(dot(Q_h(q_j), K_h(q_a)) / sqrt(d_head))。
- 在有效ANC分数前加入固定logit=0的空参照，softmax得到pi_j0、pi_ja。
- 空参照语义与几何贡献为0。不对去掉空参照后的pi重新归一化，保留关闭能力。
- 无有效ANC时不执行条件化，走原注意力路径。

实现默认：各OBJ候选共享同一组逐短语绑定w，pi则逐OBJ候选计算；不强行排除k=j的候选，不进行GT去重或额外NMS。全50候选的软绑定不等于可靠实例分解，此局限纳入报告。

### 3.3 语义融合仅作用于Q输入

每个修改层各有raw_sem_l和raw_geo_l，合计四个新标量，初值0：gamma_l=tanh(raw_sem_l)，eta_l=tanh(raw_geo_l)。

- u_j = sum_a(pi_ja q_a)。
- q_input_j = q_j + query_position_j + gamma_l rho_j u_j。
- Qtilde_j = 原Q投影(q_input_j)。

不把q_input_j写回原query残差主干。文本行Q输入保持原实现。

### 3.4 相对位置残差及等价logit修正

复用同一个原xyz_position模块PE，保持米制坐标，不尺度归一化：

- E_ki = PE(x_i − mu_k) − PE(x_i)。
- c_jk = sum_a(pi_ja w_ka)。
- D_ji = sum_k(c_jk E_ki)。

先对每条候选计算非线性PE，再加权。禁止把mu_k先平均后做PE。c只是两次加权的等价收缩，不丢弃候选或合并中心。

原Key Kbase_i = 原K投影(v_i + PE(x_i))；原Value V_i = 原V投影(v_i)。每头注意力：

A_hji = dot(Qtilde_hj, Kbase_hi)/sqrt(d_head)
       + eta_l rho_j dot(Qtilde_hj, W_K,h D_ji)/sqrt(d_head)
       + 原mask屏蔽_hji。

增量使用W_K的线性部分，不重复添加K投影偏置；原Key自身仍带原偏置。保留语义融合后的Q与几何增量的交叉作用，不能使用未融合Q近似替代。

保留原softmax、Value、out_proj、残差、norm和后续更新。原mask硬屏蔽、全屏蔽行回退和padding规则不变。几何无法恢复已被硬屏蔽的区域，不能据此宣称具备任意漏检纠错能力。

### 3.5 分块、开关与兼容

- 已确认候选块8、视觉点块1024；几何和新增logit累加FP32。块大小配置可调，不改变公式。
- 实现默认：按视觉点块计算PE增量，经原W_K线性投影并与Q收缩，累加到所需[B,H,Q,N]增量；不构造[B,Q,K,N,C]张量。不能在各视觉块内独立softmax再拼接；softmax必须覆盖全部有效视觉点。
- 为控制反向保存的PE中间激活，训练几何路径采用可复算的分块checkpoint（非reentrant），保留w/pi/Q/PE梯度；CPU比较有无checkpoint的梯度。禁止detach整个几何增量来省显存。
- baseline两个开关关闭时直接走原MultiheadAttention，避免不必要计算及随机序列变化。
- baseline四标量冻结；semantic只训练两个语义标量，几何标量冻结；full训练四个。四标量均保存进checkpoint，冻结标量不加入优化器。原共享参数初始化顺序保持一致。
- 标量值为0但开关开启时不能基于数值0跳过分支，否则标量无法获得启动梯度。无ANC的batch允许以标量乘0接入loss计算图，避免DDP未用参数；不构成新增损失或非零监督。
- 文本行和Visual反向更新的实现路径保持原结构。候选更新改变后，后续文本/Visual数值可间接受影响，不要求训练后的这些输出与基线相同。
- 多ANC数量不硬截断；沿用原文本token上限，非法或无法完整对齐的训练样本报错，不为适配新模块静默删实体。

## 4. 初始化与统一训练配置

完整复用源工程官方初始化映射：Ground3D官方纯3D权重、Qwen基座、Interactor固定资产与随机初始化规则；文本/视觉Adapter、UniVLG Decoder及角色embedding采用原seed初始化。记录真实资产路径、来源和hash，不迁移额外官方Decoder、不加载任何ScanRefer任务checkpoint。

只新训练一次Answer3轮；随后三组Joint从同一明确的Answer完成checkpoint分叉。不要用可变化的latest作为最终来源；校验Answer阶段、completed_epochs=3、hash、数据/Tokenizer协议。Joint各自新建优化器和调度器，不继承Answer AdamW动量，此行为已确认。

三组seed42（各rank沿用seed+rank），四GPU、每卡batch1、累积2，有效batch8。BF16、AdamW、clip1、原参数WD0.01；新增标量WD0。保留源数据采样和增强，不额外引入/删除增强；记录方向语言与既有坐标增强的局限，不在本次结构消融中同时改动。

| 参数组 | Answer3轮 | Joint20轮 |
|---|---|---|
| Ground3D视觉编码器 | 沿用原冻结/运行状态 | 原可训练边界，基准LR1e-5 |
| Qwen LoRA | 线性0→1e-4 | cosine 1e-4→1e-5 |
| 结构化token参数 | 基准LR1e-5 | 基准LR1e-5 |
| Interactor原可训练部分 | 基准LR1e-4 | 基准LR1e-4 |
| 文本/视觉Adapter | 不参加Answer | 各基准LR1e-4 |
| Decoder原参数 | 不参加Answer | 基准LR1e-4 |
| 启用的新增raw标量 | 不参加Answer | 随Decoder LR，WD0 |

除Qwen外保留原阶段内cosine降至10%语义，无新增warmup。源实现若存在末次更新与调度状态的端点差别，保留其实际旧轨迹并记录，不能悄悄修正造成其它组变化。

Qwen精确按实际optimizer更新编号定义，不按microbatch：
- Answer总更新T_A，编号t=0..T_A−1：lr=1e-4*t/(T_A−1)。
- Joint总更新T_J，编号t=0..T_J−1：lr=1e-5+0.5*(1e-4−1e-5)*(1+cos(pi*t/(T_J−1)))。
- 每次optimizer.step前设置该次实际LR，恢复以已完成更新计数确定下一次LR，避免scheduler.step顺序导致偏移。
- 四卡现有采样预计每轮4572个optimizer更新：Answer13716，Joint91440；由实际loader、world size和尾部累积规则计算并核实，不硬编码。预期Joint epoch10为step45720，epoch20为step91440。

损失/Matcher全部复制原ScanRefer：Answer用原SFT目标；Joint语言CE0.1，span/BCE/Dice=6/5/2，Aux0.5，ranking0.5，matcher span/BCE/Dice=2/15/6；其它focal/temperature/scope和原辅助输出层数完全保留。不得套用Reason3D CE3/OBJCE3，也不得因为新增模块重构损失或改变辅助层计数。

## 5. 工程配置、脚本及恢复

提供configs/baseline.yaml、semantic.yaml、full.yaml，复用一份实现。独立产物：outputs/answer_shared、outputs/baseline、outputs/semantic、outputs/full；日志按stage/RUN区分。共享Answer采用baseline配置中的共同语言设置；三组公共数据/语言/训练配置必须一致，配置diff白名单仅允许实验标识/路径/两个结构开关。

新增配置块anchor_relative_attention：layers=[8,9]、semantic_enabled、geometry_enabled、scalar_init=0、scalar_bound=tanh、candidate_chunk=8、visual_chunk=1024、numerical_eps=1e-6、geometry_mask_detach=true、stats_dtype=float32。模型基础配置签名与完整运行签名分开校验；保存开关/层/数学版本/配置hash和父Answer hash。

计划交付接口：
- GPU_IDS=<卡列表> bash scripts/train_answer.sh：只训练共享Answer。
- RUN=baseline|semantic|full GPU_IDS=<卡列表> bash scripts/train_joint.sh --answer-checkpoint <共享Answer最终文件>。
- 原接口CONFIG=<绝对yaml>可显式选择；若与RUN不一致立即报错，不让配置和输出目录串组。
- 两阶段保留--resume <本组checkpoint>。启动时拒绝覆盖已有checkpoint/非空metrics；错误信息给出resume或新目录用法。
- RUN=<组> GPU_IDS=<卡列表> bash scripts/evaluate_unconstrained.sh <checkpoint> <独立结果目录>。
- 同一评测入口支持--anchor-diagnostic off|disable|shuffle_geometry，默认off；提供使用示例。

GPU_IDS接受物理编号列表或UUID列表（禁止混用、重复），解析为UUID并记录物理映射；不自动选卡、不停止其他进程。保留源环境与通信检查能力；新增日志/缓存放新工程，短TMPDIR使用授权根内项目专用短路径。训练成功后不自动启动评测；每个Joint epoch保存，Answer每轮保存并显式标记完成。

同配置、同world size完整恢复：模型全部任务参数及buffers（含冻结新增标量）、各rank buffers/RNG、优化器参数组与动量、调度状态、阶段/epoch/optimizer step、父Answer来源、配置和数据指纹。恢复加载rank自己的buffer/RNG；新初始化统一同步源状态，不随机挑rank。拒绝跨baseline/semantic/full恢复、不同数学版本、数据或world size不兼容恢复。允许在评测时关闭条件化做诊断，但不能用它绕过训练恢复签名。

三组共同Decoder/Adapter初值逐值比较或分tensor哈希比较；共享初始化阶段不能因开关或四标量额外消耗RNG。校验首个epoch样本序列一致；后续训练模型不同而造成随机消耗或数值轨迹不同是正常现象，不宣称三组训练全过程随机状态相同。

## 6. 正式评测及诊断

teacher＋无格式约束generated，9483条完整验证；不teacher回退。格式/生成失败沿用原零OBJ隐藏向量的边界和解码规则，保留strictRaw完整分母，不擅自改成另一套空mask失败协议。

主指标generated strictRaw mIoU；辅指标Acc25/Acc50、unique/multiple及现有全部相关诊断。记录候选oracle、OBJ短语/格式成功率，teacher与generated差距；oracle只在正式预测/query选择完成后访问GT用于诊断，不作排序输入。

实现默认：评测使用精确无padding分片，独立rank原始结果和汇总；验证sample ID无重复/缺失、全部失败仍计入分母，主进程正常退出且全部rank汇总后才标记完整。没有达到的字段/子集需明确报错或说明源指标不支持，不能伪造0值。

独立输出evaluations/<RUN>/<checkpoint_stem>/<mode>，记录checkpoint/config/hash、GPU UUID、模式和诊断seed。保留summary及原始逐样本结果，不覆盖已有成功评测。

可选诊断均由用户指定checkpoint手动运行：
- disable：同权重、关闭两条条件化路径，保留完整生成/评分流程。
- shuffle_geometry：每个样本生成长度50的固定置换：无效候选槽位保持不变，有效候选之间使用无固定点置换，打乱几何mask对应；语义query、w/pi/rho、文本及原attention硬屏蔽不变。置换由SHA256(seed42＋稳定sample ID)派生，不能用进程随机hash；第8/9层使用相同候选槽位置换，各用自己上一层mask，rank数不影响该样本置换。
- 按原有效候选索引判定数值有效性；shuffle时将有效候选的几何中心在有效索引之间置换，避免把无效空mask引入有效候选。有效候选少于2时不打乱并记录skip。候选mask集合不变，不调用GT选择“错误”参照。
- 两种诊断结果与正式评测分开，不新增GT ANC前向通路，不自动运行。

## 7. CPU验收、实施顺序与交付

仅CPU检查与脚本语法检查，CUDA_VISIBLE_DEVICES为空；模型完整GPU前向/训练/评测不运行。优先复用既有测试和小维度替身，不在CPU重复加载数十亿参数进行无必要长测试。

1. 复制隔离与数据：配置差异白名单、36569/9483、源文件hash、资产manifest、路径/symlink隔离；旧工程无变更。实际token范围与superpoint点数对齐，训练GT不进入参照路径。
2. 前向数学：小张量显式逐query Key参考实现对比分块logit实现；覆盖多个ANC、多候选多实例、非均匀点数、padding、近空mask、无ANC、无OBJ及全屏蔽回退。含投影偏置、语义和几何同时非零的交叉项、三个独立开关组合。
3. 数值验收默认CPU FP32 atol=1e-6/rtol=1e-5，小型FP64代数测试atol=1e-10/rtol=1e-8；分块/非分块前向及梯度应一致。禁用分支走原代码时同输入/权重/dropout0应精确一致；启用但零系数时允许上述浮点容差，不承诺不同attention内核逐位一致。
4. 梯度：几何中心不反传到mask logits；w/pi/rho、ANC query、共享PE和原Q/K/V保留可用梯度。零系数首次更新允许上游新增路径梯度为0，但系数本身应可获得非零梯度；设置非零系数后检验上游梯度。无ANC批次不产生DDP未用新增标量问题；冻结分支不进优化器，四标量始终保存。
5. 训练：Answer/Joint冻结边界、四标量WD0、共同初值/RNG/样本序列；Qwen首末LR和阶段衔接、原组调度轨迹、每轮保存。用微型模型/optimizer模拟Answer→三组Joint分叉及中断恢复，验证下一步LR、动量、rank buffers/RNG一致并拒绝跨组恢复。
6. 评测：用合成分片验证9483完整分母、无padding重复、零OBJ边界、诊断不读GT、shuffle跨rank复现、skip规则与不覆盖结果。复用现有unconstrained与解析测试，不重新实现模型指标。
7. 检查所有Python语法及bash -n；记录测试命令、通过数、配置/源码hash及未执行GPU项。任何失败先诊断修复，不能仅凭静态检查宣称实际GPU显存和速度达标。

实施顺序：核验/复制隔离 → 给原baseline及计划公式建立CPU参考测试 → 在新工程扩展逐短语/点数接口和第8/9层注意力 → 分组调度/初始化签名/完整恢复 → 脚本与评测诊断 → CPU回归/语法检查 → 交付指令。任何实现不得顺带变更数据、损失、候选筛选或旧工程。

交付：本计划、新工程三份配置、共享Answer与三组Joint训练入口、指定checkpoint无约束评测/可选诊断入口、恢复示例、CPU检查报告和简要README。由用户自行启动和选择评测轮次。

## 8. 解释范围与明确的实现默认值

第1—6节中实验主干、三组配置、数据/损失、层数、绑定/几何方式、学习率/批量、路径、保存/评测边界均来自用户逐轮确认。用户最后授权剩余细节自行判断，因此本计划补齐并固定：数值分母保护、候选自引用保留、Q/K偏置处理、块内checkpoint、关闭分支冻结、脚本命名/目录、hash派生诊断置换、恢复签名和CPU容差；实施者不再自由更换这些选择。

无关系标签并不保证隐式关系学习；软候选不保证实例去重、正确多实例拆分或between联合推理；只有两个修改层且保留硬mask，纠错范围有限。首轮结果仅比较该训练预算下的结构效应，不预先承诺提升或论文新颖性。GPU速度/峰值内存须待用户实际运行验证。


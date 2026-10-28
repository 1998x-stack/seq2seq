# 论文 1409.3215 — Factory Mode 科研工程说明

原始资料：Ilya Sutskever, Oriol Vinyals, Quoc Le, Sequence to Sequence Learning with Neural Networks, 2014.
论文 HTML：https://ar5iv.labs.arxiv.org/html/1409.3215

## 1. 研究边界

论文原始训练使用约 1200 万 WMT14 英法平行句对，tokenized cased multi-bleu.pl，
以及 8-GPU C++ 并行系统。本工程默认 Helsinki-NLP/opus_books en-fr，
SacreBLEU + 自定义基本分词。不是等价的 2014 年 WMT14 指标再现。
未实现原文 5-model ensemble、SMT 1000-best rescoring、8-GPU C++ 流水线
和精确的 7.5 epoch 停止条件。原论文 34.81 BLEU 不能用来冒充当前代码的结果。

## 2. Factory 模块

| 文件 | 设计与功能 |
|---|---|
| factory/registry.py | 严格组件注册、未知类型报错，RNN 与 LSTM Cell |
| model.py | 实际使用 Cell Registry 创建 Encoder 和 Decoder |
| factory/components.py | Model / Loss / Optimizer / Scheduler / Decoder Factory |
| factory/batching.py | Random 与 LengthBucket DataLoader Factory |
| factory/experiments.py | YAML 变体、固定 manifest、训练、验证、结果汇总 |
| engine.py | 训练和验证真正接入上述工厂，而非仅写工厂类 |
| telemetry.py / visualize.py | JSONL、CSV、TensorBoard、离线 Plotly 报告 |

输入：源词序列（只源词反转）加 EOS，目标输入 BOS + 真值词，
标签真值词 + EOS。Encoder 输出状态初始化 Decoder。4-layer LSTM
会传递完整的 4 层 h/c 状态，非 Attention 模型的固定表示就是信息瓶颈。
后续的 Additive Attention 是研究扩展，非 2014 年原始模型。

## 3. 论文实验细节到配置的一一对应

| 论文证据 | 原论文 | 工程配置 | 当前状态 |
|---|---|---|---|
| §2 | 独立编码 LSTM、解码 LSTM | model.cell=lstm | 实现 |
| §2/§3.4 | 各 4 层、1000 hidden 和 embedding | model.num_layers,hidden_dim,embedding_dim | 实现 |
| §3.1 | 源 160k、目标 80k vocabulary | tokenizer.src_vocab_size/tgt_vocab_size | 实现 |
| §3.3 | 只反转 source words | model.reverse_source | 实现 |
| §3.4 | Uniform(-0.08,0.08) | model.initialization=uniform | 实现 |
| §3.4 | 无动量 SGD，初始学习率 0.7 | training.optimizer=sgd,momentum=0 | 实现 |
| §3.4 | 5 epoch 后每半轮学习率减半 | training.scheduler=paper_halving | 逐 Batch fractional epoch |
| §3.2 | 最大化按句平均 log P(T given S) | training.loss=sequence_nll | 实现 |
| §3.4 | Batch=128，clip norm=5 | batch_size,grad_clip | 实现 |
| §3.4 | 句长相近 minibatch | training.batching=length_bucket | 近似实现 |
| §3.2 | Beam 1/2/12，EOS 完成候选 | decoding.strategy=beam;beam_size | 单模型慢速参考算法 |
| §3.6 | Ensemble of 5 | 未实现 | 非当前完成范围 |
| §3.2 | 1000-best reranking | 未实现 | 非当前完成范围 |
| §3.7 | 长句性能分层 | diagnostics length_buckets | 实现 |
| §3.8 | 隐藏状态 PCA | 尚未实现 | 后续研究项 |

2014 年 WMT14 历史参照（与 OPUS 运行不得混表）：单模型正序 Beam12
26.17 BLEU，反转 Beam12 30.59；5 模型 Ensemble + Beam12 34.81。

## 4. 损失函数的数学差异

论文目标相当于先按句累计目标 NLL，再按 Batch 中句子数平均：

    L_sequence = sum_i (sum_{t not PAD} -log p(y_it)) / B

基础工程此前实现的是有效词平均：

    L_token = sum_{i,t not PAD} -log p(y_it) / N_valid_tokens

句长不同时，二者梯度贡献不同，故不能混作同一项消融。工厂实现
sequence_nll 和 token_ce 两种独立目标，测试包括长度不等的句子。
即使使用序列目标，NLL/Perplexity 评测一律按有效 Token 计数，
避免把训练损失与模型评价指标混淆。

## 5. 执行步骤

安装并验证：

    git fetch origin
    git switch feat/paper1409-factory-mode
    pip install -e ".[dev]"
    pytest -q

只看构建的参数矩阵（不会发起训练）：

    seq2seq factory plan --spec configs/factory/paper_ablation.yaml

只跑一次小型合成数据 Factory 验证：

    seq2seq factory run --spec configs/factory/smoke_factory.yaml --output runs/factory-smoke --limit 1 --max-examples 4 --split validation
    seq2seq factory visualize --output runs/factory-smoke

运行真实 OPUS Books：10 变体 x 3 seeds 共 30 组。

    seq2seq factory run --spec configs/factory/paper_ablation.yaml --output runs/paper-study --max-examples -1 --split validation
    seq2seq factory summarize --output runs/paper-study
    seq2seq factory visualize --output runs/paper-study

正式试验默认使用 validation 而不是 test 选择配置；方案锁定后
再对独立 test 统一评测，避免频繁针对测试集调参。

## 6. 研究设计限制

- 这 30 组配置不是完全全因子设计：每次只调整基线部分因素；
  modern_adamw_token_loss 同时改变优化器和目标，不能归因到单一因素。
- 不同深度和 Attention 设置的参数量/FLOPs 不同；同轮次不等于等算力。
- 解码 greedy 与 beam 更严谨的比较应共享相同训练好的 checkpoint，
  现有 declarative factory 会重复训练；下一阶段可实现纯解码实验工厂。
- 论文作者没有证明“反转后普通 RNN 就能成功训练”，原论文明确注明未验证。
- 当前 OPUS 按句随机拆分，不能声称 book-disjoint 泛化；需要可信 book_id。
- Python 参考 Beam Search 正确性优先、未做高吞吐 GPU Beam 生产优化。
- 大配置有显著算力开销，不建议在小 GPU 设备上直接启动。
- 指标只允许来自真正的训练产物，日志和图表不得编造得分。

## 7. 验收

通过测试的可观察事实：组件切换确实影响训练，初始化分布、
SGD 调度、NLL 对异长输入的归一化、Bucket 样本完整性、RNN/LSTM Beam
合法性、Manifest 冻结与拒绝数据不一致、真实日志/报告生成。
真实 OPUS 翻译效果和复现论文绝对成绩需要另行训练验证。

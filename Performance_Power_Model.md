# Title
    Using Analytical Performance/Power Model and Fine-Grained DVFS to Enhance AI Accelerator Energy Efficiency

    https://doi.org/10.1145/3669940.3707231

# Author
    Zibo Wang, Yijia Zhang*, Fuchun Wei, Bingqiang Wang, Yanlin Liu, Zhiheng Hu, Jingyi Zhang, Xiaoxin Xu, Jian He, Xiaoliang Wang, Wanchun Dou, Guihai Chen, Chen Tian* （*通讯作者）

# Institution
    南京大学 (State Key Laboratory for Novel Software Technology, Nanjing University)、鹏城实验室 (Peng Cheng Laboratory)、华为技术有限公司 (Huawei Technologies Co., Ltd)

# Submitted
    ASPLOS '25 (第30届 ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1)，2025年3月30日–4月3日，荷兰鹿特丹。DOI: 10.1145/3669940.3707231

# Tags
    AI Accelerator, Performance Model, Power Model, Fine-grained DVFS, Genetic Algorithm, Ascend NPU, Energy Efficiency, Operator-level Optimization

# Abstract
    深度学习的快速发展导致 AI 处理器能耗激增，成为制约 AI 发展的关键因素。DVFS 是功耗优化的核心方法，但由于 AI 处理器上 DVFS 控制延迟较大，以往研究只能在程序整体或子阶段粒度进行调频。最新 Ascend NPU 的毫秒级 DVFS 能力使得算子级调频成为可能。本文为每个算子构建了性能模型（平均误差 1.96%）和功耗模型（含温度因子，平均误差 4.62%），并提出基于算子分类、预处理和遗传算法搜索的 DVFS 策略。在 GPT-3 训练等任务上，AICore 功耗降低 13.44%，NPU 芯片功耗降低 4.95%，性能损失仅 1.76%。

# Problem
    1. AI 加速器功耗持续攀升（V100 300W → H100 700W），数据中心能耗占全球电力约 1% 且持续增长。
    2. 现有 DVFS 研究的调频粒度太粗：要么对整个应用设置统一频率，要么以秒级子阶段调整，无法匹配毫秒级执行的 AI 算子。
    3. NVIDIA GPU 频率切换延迟约 15ms，不支持算子级别的细粒度 DVFS。
    4. 现有性能建模多为黑盒/经验模型，缺乏对频率-性能关系本质的白盒理解；现有功耗模型忽略温度对漏电流的影响。

# Related Work
    **性能建模**：分为分析模型（如 CRISP）和统计模型（基于硬件计数器的机器学习方法），但均未深入分析 DVFS 对算子执行周期的本质影响。
    **功耗建模**：已有大量研究（AccelWattch 等），但普遍未考虑温度对芯片功耗（亚阈值漏电流）的依赖关系。
    **DVFS 策略**：多集中在 NVIDIA/AMD GPU 平台，控制粒度为程序运行时或子阶段，无人在 Ascend NPU 上做过算子级 DVFS 研究。仅有模拟层面的工作指出了细粒度 DVFS 的重要性。

# Key Contribution
    1. **白盒性能模型**：通过时间线分析证明算子的执行周期数是频率的凸分段线性函数，建模平均误差 1.96%（5000+ 算子，6 个频率点验证）。
    2. **温度感知功耗模型**：在传统 DVFS 功耗模型中引入温度依赖项，建模平均误差 4.62%。
    3. **算子级 DVFS 策略**：结合算子分类（memory-bound → 低频 LFC，compute-bound → 高频 HFC）、预处理与遗传算法搜索，实现端到端能效优化。
    4. **首个 Ascend NPU 平台 DVFS 实验研究**：利用 1ms 级 SetFreq 算子实现实际硬件上的毫秒级 DVFS 控制。

# Strengths
    1. **系统性强**：涵盖建模→策略生成→硬件验证的完整端到端流程。
    2. **理论深度**：白盒分析揭示频率-周期数的凸分段线性关系，具有可解释性和可推广性。
    3. **实验扎实**：在 GPT-3、BERT、ResNet50/152 等多个真实工作负载上验证，数据充分（Table 3 提供了不同性能损失目标下的详尽对比）。
    4. **实用价值高**：模型快速评估策略（毫秒级 vs 实测 11s/次），5 分钟可评估 20000 种策略。
    5. **可推广性讨论充分**：性能模型基于通用存储层次抽象（L1/L2/HBM），功耗模型基于物理原理而非硬件细节。
    6. **对推理场景的初步探索**：Llama2 推理实验显示 SoC 功耗降低 11.26%，展示了方法的延展潜力。

# Weaknesses
    1. **硬件绑定性**：仅在 Ascend NPU 上验证，未在 GPU/TPU 上实测，通用性声明缺乏跨平台实验支撑。
    2. **DVFS 域受限**：当前仅能对 AICore 调频，HBM/AICPU 等 uncore 组件（平均占 ~80% SoC 功耗）无法调频，整体节能空间有限（SoC 功耗仅降 ~5%）。
    3. **不适用于乱序处理器**：性能模型基于顺序执行假设，无法推广到具有复杂乱序执行机制的处理器（如 CPU）。
    4. **策略生成部分平台依赖性强**：算子分类和预处理依赖 CANN Profiler 的输出格式，迁移到其他平台需重新适配。
    5. **推理场景探索不足**：仅做了简单的全局降频实验，未结合复杂推理调度系统做深入优化。FA4 = warp_0.5_gemm + warp_0.5_softmax
    6. **遗传算法本身的局限**：未与其他优化算法（如贝叶斯优化、强化学习）做对比。

# Contents
    1. Introduction — 背景、动机与贡献概述
    2. Background — DVFS 原理、加速器存储层次
    3. Overview — 端到端能效优化流程
    4. Performance Modeling — 白盒时间线分析，凸分段线性函数建模
    5. Power Modeling — 温度感知功耗模型构建
    6. DVFS Strategy Generation — 算子分类、预处理、遗传算法搜索
    7. Evaluation — GPT-3/BERT/ResNet 实验结果与对比实验
    8. Discussion — 模型-free 方法对比、DVFS 域限制、通用性、推理场景
    9. Related Work — 性能建模、功耗建模、DVFS 策略相关研究
    10. Conclusion

# Value

    **V = innovation × effectiveness × scope**

    - **Innovation（创新性）：高**。首次在真实硬件上实现算子级毫秒 DVFS 控制；白盒分析揭示频率-周期的凸分段线性关系是新颖的理论贡献。
    - **Effectiveness（有效性）：高**。性能模型误差 1.96%，功耗模型误差 4.62%；在 GPT-3 上 AICore 功耗降低 13.44%，性能损失仅 1.76%，效果显著。
    - **Scope（适用范围）：中**。目前限于 Ascend NPU 平台且仅覆盖训练场景，但理论框架具有向 GPU/TPU 推广的潜力。

    **综合评价：V = 高**

    - **IF**：ASPLOS 是计算机体系结构/系统领域的顶级会议（CCF-A 类），不适用传统期刊 IF，但学术影响力等效于高 IF 期刊。
    - **JCR**：不适用（会议论文）。ASPLOS 在体系结构领域的地位相当于 JCR Q1 顶级期刊。

# Key Insights

    1. **算子执行周期是频率的凸分段线性函数**：这一发现源于对 core 计算与 load/store 访存时间的白盒分析——随频率升高，计算时间线性减少但访存时间不变，二者的 max 关系形成分段线性特征。这为性能建模提供了理论基础而非纯经验拟合。

    2. **温度是功耗建模不可忽视的因素**：芯片温度通过亚阈值漏电流影响功耗，传统模型忽略此项导致误差偏大。引入温度项后模型精度显著提升。

    3. **毫秒级 DVFS 是实现算子级能效优化的关键使能技术**：Ascend NPU 的 1ms 调频延迟 vs NVIDIA V100 的 15ms 延迟，这一量级差异使得逐算子设频成为可能。对比实验表明，调频间隔从 1ms 增大到 100ms 或 1s 后，节能效果明显下降。

    4. **Memory-bound 算子是降频的最佳目标**：这类算子的执行时间主要由访存决定，降低核心频率几乎不影响性能但能显著降低功耗（如 Gelu 算子仅损失 2% 性能即可换取 5%+ 功耗收益）。

    5. **模型驱动 vs 无模型方法的效率差距巨大**：有模型情况下毫秒级评估一个策略，5 分钟可遍历 20000 种方案；无模型需实测每个策略（11s/次），同等时间仅能评估 30 个——差距达 3 个数量级。

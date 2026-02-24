# 1. DartQuant剩余问题分析：
上一个PDF我们讨论了DartQuant的一个基本假设：**激活值分布都是laplacian distributed的**。然后我们引入了SWD loss来自动的衡量*任意均值为0的分布和均匀分布之间的距离*。

但是还有一些核心的问题没有解决：
1. 为什么要target均匀分布？为什么不能target高斯分布？
2. 为什么$R_3,R_4$要使用随机的Hadamard矩阵？

## 1.1 均匀假设：
上个PDF里面我回答（巧妙地避开了）问题1，我当时的回答是：“因为我们想要通过让激活值均匀分布，从而最大化信息熵来最小化量化误差”。这个理论上是没什么问题的，但是我们可以看到在LLM里面，有很多dead connections(激活值为0)。也就是说现实和理论是相违背的，尽管理论上均匀分布是最好的，但是现有的LLM激活值分布不能满足理想情况。**那为什么我们不能像QLoRA那样假设激活值分布就是高斯分布的呢？理论上把Laplace分布的激活值转换成高斯分布，比转换到均匀分布容易多了。**

## 1.2 Random Hadamard矩阵：

对于问题2，作者的回答是：$R_3$是作用于$RoPE$之后，而$RoPE$是旋转位置编码，他是位置敏感的，且不是普通的线性变化。我们无法把$R_3$穿过$RoPE$融合在前面的线性层权重里面。$R_4$是作用与$SiLU/SwiGLU$激活函数之后。非线性激活函数无法让$R_4$像$R_{1,2}$那样作用在前面的权重。所以$R_{3,4}$必须在*推理阶段*在线的计算矩阵乘法。

Hadamard矩阵具有理论最小的*互不相干性*,

$$\mu(H) = \max_{i,j} |H_{ij}| = 1/\sqrt{d}$$

以上是从矩阵元素视角考虑，当然更直观的理解是从向量内积（基变换）视角考虑：

$$\mu(H) = \max_{i,j} |\langle h_i, e_j \rangle| = \frac{1}{\sqrt{d}}$$

这意味着Hadamard矩阵能够把原来在基态方向的数值均匀分散到所有维度。但实际上，只有当数据的协方差矩阵接近于 $\sigma^2 I$（各向同性）时，*互不相干性*才能最优发挥作用。

但是$RoPE$在$R_3$处，输入数据经过了RoPE之后，被引入了块对角相关性。

**Definition 1(RoPE)**:

$$RoPE(\mathbf{x},m)=\begin{pmatrix}\cos(m\theta) & -\sin(m\theta) \\ sin(m\theta) & \cos(m\theta)\end{pmatrix}$$


随机的Hadamard本质上是对维度的随机加减组合，如果块对角之间存在这种强相关性的时候，随机的加减法可能导致相关维度上的Outlies叠加，反而增大了Outliers。

**Question 2:** 那为什么要用随机的Hadamard矩阵？没有别的支持在线推理办法了么？

> 因为你需要在推理阶段在线的实时的计算矩阵乘法，所以你的$R_{3,4}$时间复杂度肯定要越快越好，要不然就违背量化的本质和初衷了。

# 2. 改进方案：
这里介绍*可学习的旋转矩阵*和*针对于高斯分布的SWD Loss*

## 2.1 SIRB 框架：谱初始化实数分块蝴蝶变换(Spectral-Initialized Real Block Butterfly)

为了解决$R_3,R_4$无法融合导致的在线计算效率问题，同时并且随机矩阵的次优越性，我们提出使用可学习的旋转矩阵进行优化。

> **符号约定（Symbol Convention）：** 设隐藏层维度为 $d$。蝶形旋转 $R_3 \in \mathbb{R}^{d \times d}$ 直接作用于完整的 $d$ 维激活向量。定义 $K = \log_2 d$ 层蝶形操作。

### 2.1.1 RoPE分析
设 $A \in \mathbb{R}^{L \times d}$ 为输入激活矩阵，其中 $L$ 为序列长度，$d$ 为隐藏层维度。
我们将 $A$ 视为 $L$ 个行向量的集合：$A = [\mathbf{a}_1, \dots, \mathbf{a}_L]^T$。

对于第 $m$ 个位置的 token $\mathbf{a}_m$，根据Definition 1, RoPE 定义为：

$$\text{RoPE}(\mathbf{a}_m) = \mathcal{R}_m \mathbf{a}_m$$

其中 $\mathcal{R}_m \in \mathbb{R}^{d \times d}$ 是一个块对角矩阵（Block Diagonal Matrix）：

$$\mathcal{R}_m = \text{diag}\left( R_m^{(0)}, R_m^{(1)}, \dots, R_m^{(d/2-1)} \right)$$

每一个 $2 \times 2$ 的子块 $R_m^{(k)}$ 对应第 $k$ 个频率带：

$$R_m^{(k)} = \begin{bmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{bmatrix}, \quad \theta_k = 10000^{-2k/d}$$

**Remark 5**： $\text{RoPE}(A)$ 的第 $m$ 行是 $\mathbf{a}_m^T \mathcal{R}_m^T$。它保持了每个 $2 \times 2$ 子空间的模长不变（正交变换），但改变了其方向。

### 2.1.2 RoPE带来的问题 & 蝶形变换的动机
因为我们关心的是量化难度，量化难度取决于在给定数据集上特征通道的整体分布范围。因此，我们要看$\text{RoPE}(A)$在整个序列$L$上的**经验协方差矩阵(Empirical Covariance Matrix)**。

设$\tilde{\mathbf{a}}$是RoPE后的随机变量。其协方差矩阵其协方差矩阵 $\Sigma_{\text{rope}} \in \mathbb{R}^{d \times d}$ 为：

$$\Sigma_{\text{rope}} = \frac{1}{L} \sum_{m=1}^L (\mathcal{R}_m \mathbf{a}_m) (\mathcal{R}_m \mathbf{a}_m)^T = \frac{1}{L} \sum_{m=1}^L \mathcal{R}_m (\mathbf{a}_m \mathbf{a}_m^T) \mathcal{R}_m^T$$

固定一个频率索引 $k$（其余频率带类似）。**假设**输入在不同位置上的局部协方差近似不变（平稳性假设，stationarity assumption），即 $\mathbb{E}[\mathbf{a}_m^{(k)}{\mathbf{a}_m^{(k)}}^T] \approx \Sigma_{\text{in}}^{(k)}$ 对所有 $m$ 成立，其中 $\Sigma_{\text{in}}^{(k)} = \begin{bmatrix} \sigma_1^2 & \rho \\ \rho & \sigma_2^2 \end{bmatrix}$。

**RoPE第$k$个子块协方差矩阵的对角元素**:
经过代数推导（此处略去繁琐的三角恒等变换中间步骤，详细见附录），第$k$个子块协方差 $\Sigma_{\text{rope}}^{(k)}$ 的对角元素（即通道方差）为：

$$\text{Var}(\tilde{a}_{2k}) \approx \underbrace{\frac{\sigma_1^2 + \sigma_2^2}{2}}_{\text{均值项}} + \underbrace{\frac{\sigma_1^2 - \sigma_2^2}{2} \cdot \frac{1}{L}\sum_{m=1}^L \cos(2m\theta_k)}_{\text{震荡项}} \tag{*}$$

我们现在对公式(*)进行极端值分析：
1. **High Frequency (Small $k$, $\theta_k \approx 1$)**:
   $\cos(2m\theta_k)$ 在 $[-1, 1]$ 间快速震荡。当 $L$ 足够大时，$\frac{1}{L}\sum \cos \to 0$。
   $$\text{Var}_{\text{high}} \approx \frac{\sigma_1^2 + \sigma_2^2}{2}$$
   这意味着在高频部分，RoPE 起到了完美的混合作用，方差被通过旋转“抹平”了，接近于两者的平均值。

2. **Low Frequency ($k \to d/2$, $\theta_k \to 0$)**:
   $\theta_k \to 0 \implies \cos(2m\theta_k) \approx 1$。
   $$\text{Var}_{\text{low}} \approx \frac{\sigma_1^2 + \sigma_2^2}{2} + \frac{\sigma_1^2 - \sigma_2^2}{2} = \sigma_1^2$$
   这意味着在低频部分，RoPE 几乎不旋转，原始维度的方差差异被完全保留了下来。

**结论**：$\Sigma_{\text{rope}}$ 的对角线元素（Variance），随着索引 $k$ 增加（频率降低），方差从“均匀均值”逐渐演变成“原始极端值”。这种系统性的不均匀分布正是我们需要处理的核心难点。

**蝶形拓扑的优势**：
标准的 $d$ 维蝶形拓扑在前几层（小步长）会自然地配对相邻维度，而在高层（大步长）实现跨频率带的方差迁移。这种分层混合结构天然适配 RoPE 的频率带结构——低层处理子空间内部，高层处理跨频率带的能量迁移。

SIRB保留了这种蝶形拓扑的高效混合能力（$O(N \log N)$），但对节点进行了增强。

### 2.1.3 SIRB 核心定义与拓扑

#### 1. 数学定义与拓扑结构 (Definitions & Topology)
**1.1 问题定义**
设 $X \in \mathbb{R}^{B \times L \times N}$ 为神经网络中的激活张量（例如 Key/Value Cache 或 MLP 输入），其中 $N$ 为隐藏层维度（Hidden Dimension）。我们的目标是寻找一个正交旋转矩阵 $R \in O(N)$，使得旋转后的 $Y = XR^T$ 具有最小的离群值（Outliers）和最优的量化适应性。

**1.2 拓扑结构：Block Butterfly**
为了克服标准 Butterfly ($SO(2)$) 自由度低的问题，同时避免 $O(N^2)$ 的全矩阵计算，我们定义 Block Butterfly。我们将维度 $N$ 划分为 $N/4$ 个块（Block），每个块包含 4 个相邻元素。基本算子 $B^{(k)}$ 不再是平面旋转，而是 $SO(4)$ 超空间旋转。

- **输入划分**：$x = [x_1, x_2, \dots, x_{N/4}]$, where $x_i \in \mathbb{R}^4$.
- **层级变换**：第 $l$ 层的变换 $R_l$ 由块对角矩阵与置换矩阵 $P_l$ 组成：
  $$R_l = P_l \left( \bigoplus_{j=1}^{N/4} B_{l,j} \right) P_l^T$$
  其中 $B_{l,j} \in SO(4)$ 是第 $l$ 层第 $j$ 个独立的可学习块。

#### 2. 自由度 (DoF) 优势分析
这是 SIRB 优于复数方案的核心数学依据。
- **复数 $SU(2)$ 方案**：等价于实数 $4 \times 4$ 矩阵受到 Cauchy-Riemann 般的强约束：
  $$M_{\mathbb{C}} = \begin{pmatrix} A & -B \\ B & A \end{pmatrix}, \quad A,B \in \mathbb{R}^{2 \times 2}$$
  $$\text{DoF}_{SU(2)} = \dim(SU(2)) = \mathbf{3}$$

- **SIRB $SO(4)$ 方案**：解除上述约束，允许在 $SO(4)$ 流形上自由优化。
  $$\text{DoF}_{SO(4)} = \frac{4(4-1)}{2} = \mathbf{6}$$

**结论**：在相同的 FLOPs（$4 \times 4$ 矩阵乘法）下，SIRB 拥有 200% 的参数搜索空间，能够捕捉复数运算无法表达的各向异性特征。

### 2.1.4 核心数学证明：为什么 DFT 初始化优于 Hadamard？
为了回应你关于“DFT 和 $\theta=\pi/4$ 差不多”的质疑，我们从能量平滑和相位多样性两个角度证明 DFT 的必要性。

#### 1. 能量平滑定理 (Spectral Smoothing Theorem)
**定理**：对于单点离群值信号（Kronecker delta），DFT 变换能达到理论最优的波峰因子（Crest Factor）降低效果。
**证明**：假设输入 $x$ 存在极端离群值，建模为 $x = A \cdot e_k$（仅第 $k$ 个位置有值 $A$）。
$$CF_{in} = \frac{\|x\|_\infty}{\|x\|_2} = \frac{A}{A} = 1 \quad (\text{最差情况，极度尖锐})$$
应用归一化 DFT 矩阵 $F_N$：
$$y = F_N x \implies y_m = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} x_n e^{-i \frac{2\pi mn}{N}} = \frac{A}{\sqrt{N}} e^{-i \frac{2\pi mk}{N}}$$
观察输出幅度：
$$|y_m| = \frac{A}{\sqrt{N}}$$
计算输出波峰因子：
$$CF_{out} = \frac{\max |y_m|}{\sqrt{\sum |y_m|^2}} = \frac{A/\sqrt{N}}{A} = \frac{1}{\sqrt{N}}$$
**结论**：DFT 将能量从集中在 1 个点完全均匀地分摊到 $N$ 个点。这是线性变换能达到的物理极限。

#### 2. 相位多样性 (Phase Diversity)
Hadamard 矩阵是 DFT 在 $N=2^k$ 且相位量化为 $\{0, \pi\}$ 时的特例。
- **Hadamard**: 旋转角固定为 $\theta = \pi/4$。这是一种**“盲目”的等量混合**。
- **DFT**: 旋转角 $\theta_k = 2\pi k / N$ 覆盖整个单位圆。

**与 RoPE 的联系**：LLM 中的 RoPE (Rotary Positional Embedding) 将特征按频率 $\theta_i = b^{-2i/d}$ 进行旋转。
- 使用 Hadamard 初始化，相当于强行用单一频率去拟合多频率特征，这会破坏 RoPE 的位置结构。
- 使用 DFT 初始化，相当于在初始状态下提供了一个全频段的滤波器组。梯度下降只需要微调频率匹配，而不是从头学习频率结构。

### 2.1.5 完整算法流程 (The Algorithm)

#### 阶段 1: 构造与初始化 (Construction)
- **输入**: 隐藏层维度 $N$。
- **生成复数种子**: 生成标准 $N/2$ 点 FFT 矩阵 $F \in \mathbb{C}^{N/2 \times N/2}$。
- **实数映射**: 使用同构映射 $\mathcal{T}$ 将 $F$ 转换为 $N \times N$ 实数矩阵 $R_{init}$。
  $$\mathcal{T}(z) = \begin{pmatrix} \text{Re}(z) & -\text{Im}(z) \\ \text{Im}(z) & \text{Re}(z) \end{pmatrix}$$
  (注：这里是将每个复数元素扩展为 $2\times2$，整体维度变为 $N \times N$)。
- **块投影 (Block Projection)**:
  由于 $R_{init}$ 是稠密的，我们需要将其“投影”到 Block Butterfly 的参数空间作为起点。
  - **方法**：直接对 Block Butterfly 网络进行几步蒸馏 (Distillation)。固定 $R_{init}$ 作为 Teacher，随机初始化的 Block Butterfly 作为 Student。
  - **Loss**: $\| \text{Net}(I) - R_{init} \|_F^2$。
  - 收敛后的参数即为 SIRB 的初始化参数 $\Theta_0$。

#### 阶段 2: 约束释放训练 (Unconstrained Training)
- **输入**: 校准数据集 $X_{calib}$。
- **参数**: Block Butterfly 网络参数 $\Theta = \{ B_{l,j} \}_{l,j}$。
- **解锁**: 解除复数约束，允许 $B_{l,j}$ 在 $SO(4)$ 空间自由更新。
- **前向传播**:
  $$Y = \text{BlockButterfly}(X; \Theta)$$
- **损失函数**:
  $$L = L_{\text{quant}}(Y) + \lambda \cdot L_{\text{orth}}(B)$$
  - $L_{\text{quant}}$: 推荐使用 SWD (Sliced Wasserstein Distance) 或 Kurtosis (峭度) 损失，促使 $Y$ 服从高斯分布。
  - $L_{\text{orth}}$: 软正交约束，确保旋转性质。
    $$L_{\text{orth}} = \sum_{l,j} \| B_{l,j}^T B_{l,j} - I_4 \|_F^2$$

#### 阶段 3: 推理 (Inference)
- **R4 (权重矩阵)**: 训练结束后，直接计算 $R_{final}$ 并与权重合并：$W' = W R_{final}^T$。
- **R3 (激活矩阵)**: 保持 Block Butterfly 结构。利用 torch.bmm (Batch Matrix Multiply) 并行执行 $4 \times 4$ 块乘法。

### 2.1.6 梯度下降与几何优化过程
为了保证数学严谨性，我们需要处理 $B \in SO(4)$ 的流形优化。虽然软约束 ($L_{orth}$) 在工程上够用，但严谨的更新方式应沿切空间 (Tangent Space) 进行。

#### 1. 李代数更新 (Lie Algebra Update)
$SO(4)$ 的李代数 $\mathfrak{so}(4)$ 由 $4 \times 4$ 反对称矩阵 (Skew-symmetric matrices) 组成。
**优化步骤**:
1. **计算欧氏梯度**: 计算 Loss 对 $B$ 的常规梯度 $\nabla_B L$。
2. **投影到切空间**: 将梯度投影到反对称空间得到黎曼梯度 $G$。
   $$G = \nabla_B L \cdot B^T - B \cdot (\nabla_B L)^T$$
3. **更新 (Retraction)**: 使用 Cayley 变换或指数映射更新 $B$。
   $$B_{new} = \text{Cayley}(-\eta G) \cdot B_{old}$$
   $$\text{Cayley}(A) = (I - A/2)^{-1} (I + A/2)$$
**注**：这种更新方式保证 $B_{new}$ 永远严格保持正交，无需 $L_{orth}$ 惩罚项。

### 2.1.7 具体算例 (Concrete Example: N=4)
让我们手动推导一个 $N=4$ 的最小 SIRB 单元，展示它如何比复数方案更强。

**设定**: 输入向量 $x \in \mathbb{R}^4$。

- **复数方案 (Complex Butterfly)**:
  必须是 $SU(2)$ 形式。例如，绕 $z$ 轴旋转 $\phi$：
  $$B_{\mathbb{C}} = \begin{pmatrix} \cos\phi & -\sin\phi & 0 & 0 \\ \sin\phi & \cos\phi & 0 & 0 \\ 0 & 0 & \cos\phi & -\sin\phi \\ 0 & 0 & \sin\phi & \cos\phi \end{pmatrix}$$
  **限制**: 你会发现 $x_1, x_2$ 的混合方式 **必须** 和 $x_3, x_4$ 的混合方式完全一样！(Block Diagonal constraint implied by complex multiplication).

- **SIRB 方案 (Real Block)**:
  我们从 DFT 初始化开始，但允许梯度下降打破上述对称性。最终可能收敛到：
  $$B_{\text{SIRB}} = \begin{pmatrix} \cos\alpha & -\sin\alpha & \epsilon_1 & \epsilon_2 \\ \sin\alpha & \cos\alpha & \epsilon_3 & \epsilon_4 \\ \dots & \dots & \cos\beta & -\sin\beta \\ \dots & \dots & \sin\beta & \cos\beta \end{pmatrix}$$
  **关键点**:
  - $\alpha \neq \beta$：允许实部和虚部以不同的速率旋转（处理各向异性）。
  - $\epsilon \neq 0$：允许实部和虚部之间发生“串扰”（Cross-talk），这是复数乘法绝对禁止的，但对于神经网络特征混合可能非常重要。

## 2.2 针对高斯分布的 SWD Loss

回到 1.1 节的问题：为什么要 target 均匀分布？我们论证了对于 LLM 中存在大量 dead connections 的情况，将 Laplace 分布的激活值转换为高斯分布比转换为均匀分布更自然。

**Proposition 16（高斯量化的理论动机）：** 设量化器为均匀量化器（uniform quantizer），输入分布为 $p(x)$。对于 $b$-bit 量化，均方量化误差（MSQE）的高分辨率近似为：

$$D \approx \frac{\Delta^2}{12}, \quad \Delta = \frac{x_{\max} - x_{\min}}{2^b}$$

对于均匀分布，这已经是最优的。但在实际LLM中，由于dead connections的存在，将分布完全推到均匀分布需要的变换过于剧烈（ill-conditioned），而推到高斯分布是一个更温和的目标，同时配合非均匀量化器（如NF4）仍然可以获得低量化误差。

**Definition 17（Gaussian SWD Loss）：** 给定一批激活值 $\{x_i\}_{i=1}^n$（零均值），将其排序得到 $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$。目标高斯分位数为：

$$q_i = \Phi^{-1}\left(\frac{i - 0.5}{n}\right) \cdot \hat{\sigma}, \quad \hat{\sigma} = \sqrt{\frac{1}{n}\sum_i x_i^2}$$

其中 $\Phi^{-1}$ 是标准正态分布的逆 CDF。Gaussian SWD Loss 定义为：

$$\mathcal{L}_{\text{G-SWD}} = \frac{1}{n} \sum_{i=1}^n \left( x_{(i)} - q_i \right)^2$$

**Remark 18：** 与 DartQuant 的 uniform SWD 相比：
- Uniform SWD 的目标分位数是等间距的：$q_i^{\text{unif}} = x_{\min} + (x_{\max} - x_{\min}) \cdot \frac{i-0.5}{n}$
- Gaussian SWD 的目标分位数在中心密集、两端稀疏，自然适应了 LLM 激活值的尖峰分布
- $\hat{\sigma}$ 的使用确保了 loss 是尺度不变的：我们只要求分布的**形状**接近高斯，而非匹配某个特定的均值和方差

**Remark 19（与 QLoRA 的联系）：** QLoRA 的 NF4 量化方案正是基于高斯假设设计的非均匀量化 bin。我们的 Gaussian SWD Loss 可以看作是在**旋转阶段**就主动将分布塑造为高斯形状，为后续的高斯感知量化（如 NF4）提供更好的输入条件。这两者是互补的：一个优化量化器设计，一个优化输入分布。
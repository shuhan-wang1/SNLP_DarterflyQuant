# DarkQuant
## Preliminaries
LLM里的Linear layer可以表示为$Y=XW^T$, $X\in \mathbb{R}^{T\times C_\text{in}}$ denotes input activvation, $W\in \mathbb{R}^{C_\text{out}\times C_\text{in}}$ denote weight matrix. We can introduce a rotational (orthogonal) matrix $R\in \mathbb{R}^{C_\text{in}\times C_\text{in}}$ such that $Y=(XR)(R^\top W^\top)$ where $RR^\top=I$.

### 经典Transformer Block(LLamma)
![1769168502148](image/notes_darkquant/1769168502148.png)

### DartQuant后的Transformer Block (Llamma)
![1769168547723](image/notes_darkquant/1769168547723.png)
## 创新点
Propose an efficient distribution-aware rotational calibration method. DarkQuant, reduces the complexity of rotational optimization by constraining the distribution of the activations after rotation.

**Whip Loss**: Optimizes activation distribution, making it more uniform and reducing the impact of outliers.


**QR-Orth optimization scheme**： Replace expensive alternating optimization with a more efficient solution.

**Implementation and deployment:** Complete rotational calibration for *70B*  model on a single 3090 GPU(只用3小时).

## 当前问题?

### 激活outliers:
LLM后训练量化，activation pose a greater challenge than weights due to the frequent presence of *extreme outliers* which degrades models accuracy.

*Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. Affinequant: Affine transformation quantization for large language models. In The Twelfth International Conference on Learning Representations, 2024.*

这篇文章说 *rotation matrices and affine transformations* are highly effective in reducing outliers in activations, significantly improving quantization performance.

### 当前的旋转矩阵实现，为什么需要新的实现方法：
旋转矩阵可逆，保留vector norms,可以无缝插入模型框架而不用增加额外的推理cost. 随机的Hadamard rotation可以提升表现，但并不是最佳的。SpinQuant展现出旋转矩阵可以极大的提升后训练量化后的模型表现。

当前的方法SpinQuant和OSTQuant把旋转矩阵作为深度学习模型参数然后进行端到端的微调，但极大的提升了GPU显存消耗还有算力。我们可以看到DarkQuant的训练时间一直是最短的，然后accuracy是comparable甚至是更高的。

![1769166602593](image/notes_darkquant/1769166602593.png)

旋转矩阵是非常难优化的，因为我们要保留他的正交性，需要使用例如Cayley or Riemannian SGD方法。与此同时，用小样本进行端到端微调容易过拟合。为了缓解过拟合，我们可以用$R$把激活矩阵$X$变成一个更适合量化的激活矩阵。

## Method:

![1769170936353](image/notes_darkquant/1769170936353.png)

### Rotational distribution calibration（介绍框架，为什么要旋转矩阵）
因为Outliers是activation quantization loss的主要来源，所以我们的objective是约束旋转激活分布从而减小outliers的数量。

$$\min_R\sum_{i=1}^{c_\text{in}}\mathbb{I}(|(Rx)_i|>\tau)$$

$\tau$ is threshold to identify outliers。这个objective function是不可微的，所以我们需要用approximation来calibrate。*方差用于优化目标不可行，因为对称分布的激活值的variance of activations typically corresponds to a constant multiple of the activation vector’s norm square* 

However, since the rotated activations are already close to a Gaussian distribution with relatively few outliers, optimizing with kurtosis is slow (as shown in Figure 7a). Therefore, there is an urgent need for a better optimization objective to constrain the activation distribution.

### Whip loss function（怎么把原来拉普拉斯分布的激活值旋转到均匀分布）
为了减少outliers, 作者提出了一种新的受约束目标函数，用来约束激活值分布到一个均匀分布上，从而来减少旋转激活后outliers的数量

![1769192918814](image/notes_darkquant/1769192918814.png)

![1769192959029](image/notes_darkquant/1769192959029.png)

**重要假设**: 假设激活token是Laplace分布的mean 0, scale parameter $b$, 所以 PDF是

$$f(x) = \frac{1}{2b}\exp(-\frac{|x|}{b})$$

我们要用CDF来吧$x\tilde Laplace(0,b) \to x \tilde [-\tau, \tau]$我们需要

$$U_X(x)=2\tau[\int_{-\infty}^x \frac{1}{2b}\exp(-\frac{|x|}{b})dt-\frac12]\\
= \begin{cases}
\tau[\exp(\frac{x}{b})-1], \quad x\le 0\\
\tau[1-\exp(-\frac{x}{b})], \quad x > 9
\end{cases}$$

由此可得Whip Loss:

$$\mathcal{L} = \sum_{i=1}^{c_\text{in}}\exp(-|x_i|)$$

where $\mathbf{x}=[x_1,...,x_{{C}_\text{in}}] \in \mathbb{R}^{C}_\text{in}$。本质上我们就是想要数据不要像Laplace那样挤在mean，而是让他均匀分布在$[-\tau, \tau]$上。**注意，后面文章会讲到如何把范数不变约束条件放入优化过程，这里先假设激活值必须符合范数不变**。因为$||Rx||_2=||x||2$所以如果中间值被拉开了，那outliers就得减小从而符合范数不变。
### OR-Orth optimization
我们来看一下Navie的目标，我们的目标是找一个旋转矩阵$R$，使得量化误差最小

$$\min_R \mathcal{L}(XR): \quad R^\top R=I$$

如果直接使用梯度下降

$$R'=R-\eta\frac{\partial \mathcal{L}}{\partial R}$$

因为$R$正交，那你$R'$减去梯度后很难保持原来的正交性，如果失去正交性那就不是旋转矩阵了，导致激活值范数改变，模型量化失败。传统方法是用$Cayley SGD$，在manifold上通过复杂的曲线路径更新参数，时间复杂度是$O(n^3)$

OR-Orth引入一个隐变量，$Z\in\mathbb{R}^{n\times n}$，随便一个square matrix，没有任何约束。

映射（Gram-Schimidt正交化）：在每次前向传播时，算法对$Z$进行QR分解：

$Z = R \cdot U$

R是QR分解后的严格的正交矩阵，U是缩放和剪切的，直接丢掉。然后我们只需要对$Z$关于Whip loss进行梯度下降即可，

### 完整OR-Orth算法实现：
1. 用Hadamard随机初始化一个矩阵$Z_0$
2. 把Z通过QR分解映射到一个正交矩阵
3. 计算 $XR$, 计算$\mathcal{L}(XR)=Whip(XR)$ whip loss
4. $Z \lArr Z - \eta \frac{\partial \mathcal{L}}{\partial Z}$
   - 注意我们这里要更新的是$R$不是$Z$，更新后$Z$对应的不是上一步的$R$也可能不会正交。但无所谓因为我们要的是最后的$R$，$Z$只是隐变量
   - 循环步骤2-4

### 后续的量化过程
# Appendix:
## A: 激活值真的都是Laplace分布的么？如果mean不是0怎么办？
不完美符合，但Laplace已经是很好的近似了。表19测试了Llama系列下模型的激活值基本上mean都是$1e-2$量级的，所以基本上都是mean。作者也说了需要更rigorous的学习激活值的统计分布，所以如果mean不是0那就这个假设就废了。

![1769194688478](image/notes_darkquant/1769194688478.png)

![1769194672006](image/notes_darkquant/1769194672006.png)


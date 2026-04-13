# Setups:
假设 **Int4格式** 量化：
- SWD_unif
- Whip
*量化器*: GPTQ
假设 **NF4格式** 量化：
- SWD_Gauss bnb

# Experiment 1:
两个SWD_loss (SWD_unif和SWD_gauss)和Whip loss直接对比（只替换loss function，别的训练框架，除了SWD_loss使用的是bnb量化外都不变）。在W4A4KV4下，SWD_unif表现最佳，whip其次，bnb表现最差。

结论：swd_unif牛逼

# Experiment 2:
在W4A4KV4量化下，对作用R3,R4后的激活进行NF4格式的量化是不合理，不公平的。因为Random Hadamard的作用是均匀分散能量，而不是“高斯的”分散能量。也就是说你的R3,R4 target的是均匀分布，你把激活打散成均匀分布，然后再用假设是NF4格式的量化器就说Gauss是垃圾的是不公平的。所以增加实验W4A16KV16的实验。

结论：Gauss还是垃圾，但bnb本身就不够robust，他就天然是垃圾的，他没有GPTQ robust。他没有calibration的步骤，直接round了。

# Experiment 3:
Instruct只在特定的数据集牛逼，取决于他在什么类型的数据集进行了量化。在general的ppl上，instruct总是比uninstruct垃圾。无论是量化前，量化后，无关量化器，量化格式。


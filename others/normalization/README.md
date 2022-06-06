# 全网最详细、最全面、最易懂的normalization解读
# 深度学习中的归一化问题


深度学习领域，常用的Normalization方法主要有：
- [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)（BN，2015年）
- [Layer Normalization](https://arxiv.org/pdf/1607.06450v1.pdf)（LN，2016年）
- [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf)（IN，2017年）
- [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)（GN，2018年）
- [Switchable Normalization](https://arxiv.org/pdf/1806.10779.pdf)（SN，2018年）

它们都是从激活函数的输入来考虑、做文章的，以不同的方式对激活函数的输入进行 Norm 的。

我们将输入的 feature map shape 记为[N, C, H, W]，其中N表示batch size，即N个样本；C表示通道数；H、W分别表示特征图的高度、宽度。这几个方法主要的区别就是在：

1. BN是在batch上，对N、H、W做归一化，而保留通道 C 的维度。BN对较小的batch size效果不好。BN适用于固定深度的前向神经网络，如CNN，不适用于RNN；

2. LN在通道方向上，对C、H、W归一化，主要对RNN效果明显，Transformer上的归一化也是用的LN；

3. IN在图像像素上，对H、W做归一化，用在风格化迁移；

4. GN将channel分组，然后再做归一化。
5. SN是将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

![image](7230D33DBD464A7D8F7ACE82E14E5CF1)

上图来源于论文[GroupNorm](https://arxiv.org/pdf/1803.08494.pdf)。

## BN(Batch Normalizatioon)
### BN为了解决什么问题？
1.在深度神经网络训练的过程中，通常以输入网络的每一个mini-batch进行训练，这样每个batch具有不同的分布，使模型训练起来特别困难。

2.内部协变量偏移(Internal Covariate Shift):深层神经网络中，中间某一层的输入是其之前的神经层的输出。因此，其之前的神经层的参数变化会导致其输入的分布发生较大的差异。在训练过程中，激活函数会改变各层数据的分布，随着网络的加深，这种改变（差异）会越来越大，使模型训练起来特别困难，收敛速度很慢，会出现梯度消失的问题。


### BN的主要思想
针对每个神经元，使数据在进入激活函数之前，沿着通道计算每个batch的均值、方差，‘强迫’数据保持均值为0，方差为1的正态分布，避免发生梯度消失。

具体来说，我们令第`$l$`层的净输入为`$z^{(l)}$`,神经元的输出为`$a^{(l)}$`

![image](99D4D5FDE2CD40A5A71A308094E3E2EA)

为了减少内部协变量偏移问题，就要使得净输入`$z^{(l)}$` 的分布一致，比如都归一化到标准正态分布。但是逐层归一化需要在中间层进行操作，要求效率比较高，因此复杂度比较高的白化方法就不太合适。为了提高归一化效率，一般使用标准归一化，将净输入`$z^{(l)}$`的每一维都归一到标准正态分布：

![image](D9885E2315A7417985EB7E1650A8D531)

`$E(z^{(l)})$`和 `$var(z^{(l)})$`指当前参数下， `$z^{(l)}$` 的每一维在整个训练集上的期望和方差。因为目前主要的训练方法是基于mini-batch的随机梯度下降方法，因此 `$z^{(l)}$` 的期望和方差通常用当前mini-batch样本集的均值和方差近似估计。

给定一个包含 `$K$`个样本的小批量样本集合，第 `$l$` 维度（通道）神经元的净输入 `$z^{(1, l)}, z^{(2, l)}, \ldots, z^{(k, l)}$` 的均值和方差为：

![image](FA7C61B7FADD4BF1B6082C086C77E534)

注意：这里的均值和方差是针对mini-batch中K个样本的的相同维度（通道）进行计算的。(下图中的m=batch_size=k=2)

![image](F13CF6A79D1043469F476FB87D9D7DD5)

对净输入`$z^{(l)}$`的标准归一化会使得其取值集中在0附近，如果使用sigmoid型函数时，这个取值区间刚好接近线性变换区间，减弱了神经网络的非线性性质，因此，为了使得归一化不对网络的表示能力造成负面影响，我们可以通过一个附加的缩放和平移变换改变取值区间。

![image](AF8ABA26DD1B48C2AC6E621E3C500490)

其中，`$\gamma $`表示缩放参数。`$\beta $`表示平移参数。

**注意：均值`$\mu $`和方差`$\sigma_{2}^{2}$`是网络前向传播得到的。缩放参数`$\gamma $`和平移参数`$\beta $`是网络反向梯度传播得到的。**

### BN伪代码

![image](84D2DBD6AEE34AD4A637DC4D6E19EEE4)

### BN的使用位置
全连接层或卷积层之后，激活函数之前

### BN的优势
- 允许较大的学习率
- 减弱对初始化的强依赖性，降低权重初始化的困难
- 保持隐藏层中数值的均值，方差不变，控制数据的分布范围，避免梯度消失和梯度爆炸
- BN可以起到和dropout一样的正则化效果，在正则化方面，一般全连接层用dropout,卷积层拥BN
- 缓解内部协变量偏移问题，增加训练速度

### BN存在的问题
- 每次是在一个batch上计算均值、方差，如果batch size太小，则计算的均值、方差不足以代表整个数据分布。
- batch size太大：会超过内存容量；需要跑更多的epoch，导致总训练时间变长；会直接固定梯度下降的方向，导致很难更新。

## LN(Layer Normalization)
### LN为了解决什么问题？
如果一个神经元的输入分布在神经网络中是动态变化的，比如循环神经网络（RNN），那么无法应用BN操作。因此，针对BN不适用于深度不固定的网络（RNN，sequeece长度不一致）的问题，LN针对单个训练样本进行归一化操作，即对每一个样本中多个通道（channel）进行归一化操作。

### LN的主要思想
LN针对单个样本进行归一化操作。具体来说，对于输入`$x\in \mathbb R^{N\times C\times H\times W} $`,LN对每个样本的C、H、W维度上的数据求均值和方差，保留N维度。LN中不同的输入样本有不同的均值和方差。

其均值公式：

![image](970505B1603F4C0383BA13E7EDC0EB0B)

方差公式：

![image](4BFCA4E27822423FABD5CC3BE72547AC)

**注意：LN的均值和方差是针对单个样本的所有通道进行计算的、**

![image](07CB0B649C434236B01BEE26ACE6EEBB)

与BN一样，LN也有缩放参数和平移参数

![image](85D65EE82D594484A13F0E5F04D66E07)

### LN的优势
不需要批训练，在单条数据内部机就能归一化。不依赖batch_size和输入sequence的长度，因此可以用于batch size为1和RNN中。LN用于RNN效果比较明显，但是在CNN上，效果不如BN。

## IN(Instance Normalization)
### 为什么提出IN？
IN针对图像像素做normalization，最初用于图像的风格化迁移。在图像风格化中，生成结果主要依赖于某个图像实例，feature map 的各个 channel 的均值和方差会影响到最终生成图像的风格。所以对整个batch归一化不适合图像风格化中，因而对H、W做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。如果特征图可以用到通道之间的相关性，那么不建议使用IN。

### IN的做法
IN是针对一个样本中的每个通道进行单独的归一化操作。具体来说，针对输入`$x\in \mathbb R^{N\times C\times H\times W}$`，IN对每个样本的每个通道的`$H,W$`维度的数据求均值和方差，保留`$N,C$`维度。换言之，IN旨在channel内部求均值和方差。

求均值公式：
![image](2596DA2F8343412093C54B9938A0B135)

方差公式：

![image](DB911B9E88ED44D3B6F4CDE5687D6900)

**注意：IN的均值和方差只针对单样本单通道计算**。

![image](59971C8F182D434CABDB7CCF3E9836CA)
## GN(Group Normalization)
### 为什么提出GN？
GN是为了解决BN对较小的mini-batch size效果差的问题。GN适用于占用显存比较大的任务，例如图像分割。对这类任务，可能 batch size 只能是个位数，再大显存就不够用了。而当 batch size 是个位数时，BN 的表现很差，因为没办法通过几个样本的数据量，来近似总体的均值和标准差。GN 也是独立于 batch 的，它是 LN 和 IN 的折中。

### GN的主要思想
GN将channel分组，即在channel方向做group，然后每个group内做normalization。具体来说，针对输入`$x\in\mathbb R ^{N\times C\times H\times W}$`,GN把每一个样本feature map的channle分为G组，即每组有`$C/G$`个channel，然后将这些channel中元素求均值和方差。各组GN相互独立。

求均值公式：

![image](8E922366DCDA4DAEB6EA165979DD2C4F)

方差公式：

![image](FFFE4DEB20BD4E8B874B93DD4EFC693E)

**注意：GN的均值和方差只针对单个样本G组通道内做归一化，每组包含`$C/G$`个通道。当`$G=C$`时，GN等价于IN。当`$G=1$`时，GN等价于LN。**

![image](C52425AF36314496A44BAF3B27D751CE)

## SN（Switchable Normalization）[github](https://github.com/switchablenorms/Switchable-Normalization)


Switchable Normalization将 BN、LN、IN 结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

SN具有以下三个优点：
- 鲁棒性：无论batchsize的大小如何，SN均能取得非常好的效果；
- 通用性：SN可以直接应用到各种类型的应用中，减去了人工选择归一化策略的繁琐；
- 多样性：由于网络的不同层在网络中起着不同的作用，SN能够为每层学到不同的归一化策略，这种自适应的归一化策略往往要优于单一方案人工设定的归一化策略。

![image](4D8F41DB0BD94BFC972864369F2F9FC7)

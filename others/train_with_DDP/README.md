# 多GPU启动指令  DDP模式

## DDP原理
DistributedDataParallel（DDP）支持多机多卡分布式训练

通俗理解：

DDP模式会开启N个进程，每个进程控制一张显卡上加载模型，这些模型相同（被复制了N份到N个显卡），缓解GIL锁的限制。
训练阶段，每个进程通过Ring-Reduce的方法与其他进程通讯（交换各自的梯度）
各个进程使用平均后的梯度更新自己的参数，因为每个进程下模型的初始参数、更新梯度是一样的，所以更新后模型的参数也保持一致

## DP原理
model=torch.nn.DataParallel(model)

DP模式中只有一个进程，容易受到GIL的限制。master节点相当于参数服务器，向其他卡广播参数，在梯度反向传播后，每个卡将梯度汇总到master节点，master对梯度进行平均后更新参数，再将参数发送到其他卡上。

显而易见的，这种模式会导致节点的计算任务，通讯量很重，从而导致网络阻塞，降低训练速度。

## GIL是什么？为什么DDP更快
GIL（全局解释器锁，可以参考GIL），主要的缺点就是：限制python进程只能利用一个CPU核心，不适合计算密集型的任务。使用多进程，才能有效利用多核的计算资源。DDP启动多进程，一定程度上避免了这个限制。

Ring-Reduce梯度合并：各个进程独立计算梯度，每个进程将梯度依次传给下一个进程，之后再把从上一个进程拿到的梯度传给下一个进程，循环n（进程数量）次之后，所有的进程就可以得到全部的梯度。（闭环）


## DDP相关概念
- rank：用于表示进程的编号/序号（在一些结构图中rank指的是软节点，rank可以看成一个计算单位），每一个进程对应了一个rank的进程，整个分布式由许多rank完成。
- node：物理节点，可以是一台机器也可以是一个容器，节点内部可以有多个GPU。
- rank与local_rank： rank是指在整个分布式任务中进程的序号；local_rank是指在一个node上进程的相对序号，local_rank在node之间相互独立。
- nnodes、node_rank与nproc_per_node： nnodes是指物理节点数量，node_rank是物理节点的序号；nproc_per_node是指每个物理节点上面进程的数量。
- word size ： 全局（一个分布式任务）中，rank的数量。
```
每个node包含16个GPU，且nproc_per_node=8，nnodes=3，机器的node_rank=5，请问word_size是多少？   

答案：word_size = 3*8 = 24 
```

## 启动方式
- torch.distributed.launch
- torch.multiprocessing.spawn

## torch.distributed.launch参数介绍
- nnodes 有多少台机器
- node_rank 当前是哪台机器
- nproc_per_node 每台机器有多少进程

## 单机多卡
```
# 假设我们只在一台机器上运行，可用卡数是8
python -m torch.distributed.launch --nproc_per_node 8 train.py
```

- 如果要指定使用某几块GPU可使用如下指令，例如使用第1块和第4块GPU进行训练：
```
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 train.py --device 0,3
```



## 多机多卡
- master_address: master进程的网络地址
- master_port: master进程的一个端口，默认29500，使用前需要确认端口是否被其他程序占用。
-  假设一共有两台机器（节点1和节点2），每个节点上有8张卡，节点1的IP地址为192.168.0.1 占用的端口12355（端口可以更换），启动的方式如下：
```
>>> #节点1
>>>python -m torch.distributed.launch --nproc_per_node=8
           --nnodes=2 --node_rank=0 --master_addr="192.168.0.1"
           --master_port=12355 MNIST.py
>>> #节点2
>>>python -m torch.distributed.launch --nproc_per_node=8
           --nnodes=2 --node_rank=1 --master_addr="192.168.0.1"
           --master_port=12355 MNIST.py
```
- 其中```nproc_per_node```为并行GPU的数量



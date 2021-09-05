# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

# 获取当前脚本的绝对路径
# from pathlib import Path，获取与系统路径无关的路径，比如windows:\\ ,linux:/, mac: :
FILE = Path(__file__).absolute()
# 将当前脚本的绝对路径的父路径添加到系统路径中，方便项目能找到对应模块
# as_posix():转换为str
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import time_sync, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    # thop:计算FLOPS的模块
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# 实例化日志
LOGGER = logging.getLogger(__name__)

# 检测层
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes 类别数量
        self.no = nc + 5  # number of outputs per anchor 每个anchor输出的维度
        self.nl = len(anchors)  # number of detection layers FPN的层数
        self.na = len(anchors[0]) // 2  # number of anchors 每层anchor的数量
        self.grid = [torch.zeros(1)] * self.nl  # init grid 初始化网络坐标,3个网格，对应3个FPN
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # 模型中需要保存下来的参数分为两种：一种是反向传播需要被optimizer更新的，称之为 parameter
        # 一种是反向传播不需要被optimizer更新的，称之为 buffer
        # 第二种参数需要创建tensor,然后将tensor通过register_buffer()进行注册
        # 可以通过model.buffer()返回，注册完参数也会自动保存到OrderDict()中
        # 注意：buffer的更新在forward中，optim.step只能更新nn.parameter
        # 注册a、anchor_grid为buffer，可保存到网络权重中
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        # 将anchor reshape为与网络输出shape一致
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # 检测头，输出层
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # 是否直接在预测y上反算坐标并替换
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        # 存储推理的输出结果
        z = []  # inference output
        # 分别遍历3个fpn特征层
        for i in range(self.nl):
            # x[i]表示第i层的FPN输出，m[i]表示第i层的检测头
            x[i] = self.m[i](x[i])  # conv
            # 获取特征图输出的shape:bs, 255, 20, 20
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 将x[i]的维度变为：bs, 3, 20, 20, 85
            # bs：批次大小
            # 3：当前特征图每个格子3个anchor
            # 20, 20：特征图size，640/32=20,32表示经过5次下采样操作
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 表示推理阶段
            if not self.training:  # inference
                # 如果是前向推理, 创建网格坐标
                # 判断一下网格和特征图的维度大小(20, 20, 85)是否一致
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    # 不一致，则生成特征图网格坐标
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                # 对所有输出限制范围0~1
                y = x[i].sigmoid()
                # 预测框坐标反算，公式参见https://github.com/ultralytics/yolov5/issues/471
                if self.inplace:
                    # 切片取值
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """生成特征图网格坐标"""
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    """创建模型"""
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            # 获取网络结构配置文件
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        # 定义模型
        # 输入的通道数
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 如果传入的类别和yaml文件的不一致, 以传入的nc为主
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 重写yaml的类别数量的值，以coco128.yaml的为准,重写yolov5s.yaml的nc
            self.yaml['nc'] = nc  # override yaml value
        # 如果传入了anchor, 以传入的anchor为主
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 开始定义网路结构.deepcopy():深拷贝
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # 单独对检测头处理
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 前向推理一次获取每层FPN层输出的步长
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 将基于原图的anchor 缩放为基于特征图大小的，这里的基于特征图大小的anchors在计算损失的时候有使用
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查anchor顺序是否余stride顺序一致
            check_anchor_order(m)
            self.stride = m.stride
            # 初始化检测头网络的biases
            self._initialize_biases()  # only run once
            # LOGGER.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        # 初始化网络权重
        initialize_weights(self)
        # 显示模型信息
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        """前向推理"""
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        return self.forward_once(x, profile, visualize)  # single-scale inference, train

    def forward_augment(self, x):
        """Test Time Augmentation
        对图片以固定的尺度进行缩放，翻转再送入网络模型推理
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # 对图片进行缩放
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            # 正常前向推理
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # TTA时将数据增强的图片预测 反算为基于原图的预测
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, visualize=False):
        """正常前向推理"""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_sync()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        """TTA时将数据增强的图片预测 反算为基于原图的预测"""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        """初始化检测头网络的biases，使网络在训练初期更稳定"""
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        """打印偏置"""
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """模块融合"""
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 将卷积层和bn层融合为一层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 删除原网络中的bn层
                delattr(m, 'bn')  # remove batchnorm
                # 将新的层更新到源网络中
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        """显示模型的信息，网络层信息，参数量，梯度量等"""
        model_info(self, verbose, img_size)

# d:配置网络结构的参数，为dict类型.参考yolov5s.yaml的内容
# ch:输入的通道数，3
def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # cnchors:
    # nc:类别数量
    # gd:网络的深度，不同的网络深度通过该参数进行控制
    # gw:网络的宽度，不同的网络宽度通过该参数进行控制
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # 每个格子对应3个anchor。 6/3=2
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # 每个格子输出向量的大小，3*(80+5)，即检测头输出通道数
    # 3:每个格子anchor的数量
    # 80：coco数据集，80个类别
    # 5=4+1 x,y,w,h,c
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    # 初始化列表，用来保存网络层，需要保存的网络层索引，该层输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # d['backbone'] + d['head']:将两部分的列表内容合并
    # i：遍历到的列表下标，从0开始
    # (f, n, m, args):
    # f:from,表示该层网络接收哪一层的输出作为输入，为网络层的索引，-1表示上一层
    # n:number, 表示该m模块的堆叠次数
    # m:module,模块名字
    # args：初始化m模块时的参数，通常是一些输出通道数，卷积核大小，stride大小等
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # eval():执行字符串表达式，即将字符串的引号去掉，保留内容，创建内容对应的方法
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                # eval():执行字符串表达式，即将字符串的引号去掉，保留内容
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        # 通过gd控制网络深度
        # 只有当n>1的时候，才会修改网络的深度，即csp层中的残差组件数量
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 判断模块是否属于列表中的模块
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            # c1:该层输入通道数，
            # c2：该层输出通道数
            c1, c2 = ch[f], args[0]
            # 判断是否为检测头，不是则对输出通道进行缩放，不同网络的网络宽度是不一样的，即卷积核数量不同
            if c2 != no:  # if not output
                # 通过gw控制网络宽度，如yolov5s，focus输出通道是32
                #                     如yolov5l,focus输出通道是64
                # make_divisible():保证结果可被8整除
                c2 = make_divisible(c2 * gw, 8)

            # *a:a为列表，访问列表元素
            # **a:a为字典，访问字典元素
            # 更新该层对应的参数
            # c1:该层输入通道数，从上一层的输出通道数数获得ch[f],f=-1
            # c2:该层输出通道数，通过gw控制通道数（网络宽度）
            # *args[1:]:[通道数，卷积核大小，stride，等参数] 具体参考yolo5s.yaml
            args = [c1, c2, *args[1:]]
            # 对于这些层，将堆叠的次数插入到args中
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        # 下面是根据不同的模块，调整args
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            # 将FPN的输出通道数添加到args
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # 根据前面解析的信息创建模型
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        # 显示每一层网络层的信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        # 保存该层的输出通道数，方便作为之后层的输入通道数信息
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        # img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph

import torch
from d2l import torch as d2l

# 多尺度目标检测

# 在多个尺度下，我们可以生成不同尺寸的锚框来检测不同尺寸的目标。
# 通过定义特征图的形状，我们可以决定任何图像上均匀采样的锚框的中心。
# 我们使用输入图像在某个感受野区域内的信息，来预测输入图像上与该区域位置相近的锚框类别和偏移量。
# 我们可以通过深入学习，在多个层次上的图像分层表示进行多尺度目标检测。

img = d2l.plt.imread('d2l-zh/pytorch/img/catdog.jpg')
h, w = img.shape[:2]

def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
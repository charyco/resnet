import pandas as pd
import os
import shutil
from PIL import Image


class KF_Dataset():

    def __init__(self, csv_path, data_img_path, spilt):
        super(KF_Dataset, self).__init__()
        self.spilt = spilt
        self.imgs, self.labels = load_data(csv_path, data_img_path)


    def __getitem__(self, index):
        if self.spilt == 'train':
            img = Image.open('/data/users/user2/xjm/demo/plant_dataset/plant_dataset/train/images/'+self.imgs[index]).convert('RGB')
        else:
            img = Image.open('/data/users/user2/xjm/demo/plant_dataset/plant_dataset/test/images/'+self.imgs[index]).convert('RGB')
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)

import mindspore.dataset as de
from mindspore import dtype as mstype
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.dataset.vision import ImageBatchFormat
from mindspore.dataset.vision import AutoAugmentPolicy, Inter

def create_dataset(dataset, repeat_num=1, batch_size=32, target='train', image_size=224):
    # 定义数据增强
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    scale = 32
    # type_op = C2.TypeCast(mstype.float32)
    if target == "train":
        # Define map operations for training dataset
        trans = [
            C.Resize(size=[image_size, image_size]),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomRotation(degrees=15),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            C.AutoAugment(policy=AutoAugmentPolicy.IMAGENET,interpolation=Inter.NEAREST,fill_value=0),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            # type_op
        ]
    else:
        # Define map operations for inference dataset
        trans = [
            C.Resize(size=[image_size + scale, image_size + scale]),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            # type_op
        ]
    cutmix_batch_op = C.CutMixBatch(ImageBatchFormat.NCHW, 1.0, 0.5)

    dataset = de.GeneratorDataset(dataset, ["image", "label"])
    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    
    # 设置batch_size的大小，若最后一次抓取的样本数小于batch_size，则丢弃
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if target == "train":
        dataset = dataset.map(operations=cutmix_batch_op, input_columns=["image", "label"], num_parallel_workers=8)
    # dataset = dataset.map(operations=type_op, input_columns="label", num_parallel_workers=8)
    # 设置数据集重复次数
    dataset = dataset.repeat(repeat_num)
    return dataset

label_dict = {
    'scab': 0, # 痂
    'healthy': 1, # 健康
    'frog_eye_leaf_spot': 2, # 青蛙眼叶斑病
    'rust': 3,  # 植物的锈病
    'complex': 4,  # 多并发症
    'powdery_mildew' : 5 # 白粉霉病
}
import numpy as np
def load_data(datacsv_path,dataimg_path):
    db = np.loadtxt((datacsv_path), str, delimiter=",", skiprows=1)
    labels_list=[]
    images_list=[]
    for i in range(len(db[:,1])):
        label = np.zeros(shape=(6), dtype=np.float32)
        label_str = db[i,1]
        for item in label_str.split(" "):
            label[ label_dict[item] ] = 1
        labels_list.append(label) #onhot
        image_file = os.path.join(db[i,0])
        images_list.append(image_file) 
    return images_list, labels_list



import mindspore.dataset as de
from mindspore import dtype as mstype
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.dataset.vision import ImageBatchFormat
from mindspore.dataset.vision import AutoAugmentPolicy, Inter

def create_dataset(dataset, repeat_num=1, batch_size=32, target='train', image_size=224):
    # 定义数据增强
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    scale = 32
    # type_op = C2.TypeCast(mstype.float32)
    if target == "train":
        # Define map operations for training dataset
        trans = [
            C.Resize(size=[image_size, image_size]),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomRotation(degrees=15),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            C.AutoAugment(policy=AutoAugmentPolicy.IMAGENET,interpolation=Inter.NEAREST,fill_value=0),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            # type_op
        ]
    else:
        # Define map operations for inference dataset
        trans = [
            C.Resize(size=[image_size + scale, image_size + scale]),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            # type_op
        ]
    # cutmix_batch_op = C.CutMixBatch(ImageBatchFormat.NCHW, 1.0, 0.5)

    dataset = de.GeneratorDataset(dataset, ["image", "label"])
    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    
    # 设置batch_size的大小，若最后一次抓取的样本数小于batch_size，则丢弃
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # if target == "train":
    #     dataset = dataset.map(operations=cutmix_batch_op, input_columns=["image", "label"], num_parallel_workers=8)
    # dataset = dataset.map(operations=type_op, input_columns="label", num_parallel_workers=8)
    # 设置数据集重复次数
    # dataset = dataset.repeat(repeat_num)
    return dataset
import random
import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.communication.management import get_rank, init, get_group_size
from mindspore import Tensor, Model, context, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback

import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
"""ResNet."""
import math
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)

def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    else:
        weight_shape = (out_channel, in_channel, 3, 3)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                         padding=1, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    else:
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                         padding=0, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    else:
        weight_shape = (out_channel, in_channel, 7, 7)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if res_base:
        return nn.Conv2d(in_channel, out_channel,
                         kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel, res_base=False):
    if res_base:
        return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

def _fc(in_channel, out_channel, use_se=False):
    if use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
    else:
        weight_shape = (out_channel, in_channel)
        weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
        self.bn1 = _bn(channel)
        if self.use_se and self.stride != 1:
            self.e2 = nn.SequentialCell([_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                         nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')])
        else:
            self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
            self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        self.bn3 = _bn(out_channel)
        if self.se_block:
            self.se_global_pool = ops.ReduceMean(keep_dims=False)
            self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
            self.se_mul = ops.Mul()
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            if self.use_se:
                if stride == 1:
                    self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel,
                                                                         stride, use_se=self.use_se), _bn(out_channel)])
                else:
                    self.down_sample_layer = nn.SequentialCell([nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
                                                                _conv1x1(in_channel, out_channel, 1,
                                                                         use_se=self.use_se), _bn(out_channel)])
            else:
                self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                     use_se=self.use_se), _bn(out_channel)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_se and self.stride != 1:
            out = self.e2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se_block:
            out_se = out
            out = self.se_global_pool(out, (2, 3))
            out = self.se_dense_0(out)
            out = self.relu(out)
            out = self.se_dense_1(out)
            out = self.se_sigmoid(out)
            out = ops.reshape(out, ops.shape(out) + (1, 1))
            out = self.se_mul(out, out_se)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Cell):
    """
    ResNet architecture.
    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net in layer 3 and layer 4. Default: False.
        res_base (bool): Enable parameter setting of resnet18. Default: False.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 use_se=True,
                 res_base=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        if self.use_se:
            self.se_block = True

        if self.use_se:
            self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
            self.bn1_0 = _bn(32)
            self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
            self.bn1_1 = _bn(32)
            self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        else:
            self.conv1 = _conv7x7(3, 64, stride=2, res_base=self.res_base)
        self.bn1 = _bn(64, self.res_base)
        self.relu = ops.ReLU()


        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       use_se=self.use_se,
                                       se_block=self.se_block)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       use_se=self.use_se,
                                       se_block=self.se_block)

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, use_se=self.use_se)
        self.sigmoid = nn.Sigmoid()


    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        if se_block:
            for _ in range(1, layer_num - 1):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
            layers.append(resnet_block)
        else:
            for _ in range(1, layer_num):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_se:
            x = self.conv1_0(x)
            x = self.bn1_0(x)
            x = self.relu(x)
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.relu(x)
            x = self.conv1_2(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)
        out = self.sigmoid(out)
        return out


def resnet101(class_num=1001):
    """
    Get ResNet101 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet101 neural network.

    Examples:
        >>> net = resnet101(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)

#计算准确率
def count_accuracy(gt, pre):
    cnt=0.0
    for i in range(len(gt)):
        flag=0
        for j in range(6):
            if pre[i,j]<0.5 and gt[i,j]==1:
                flag=1
            elif pre[i,j]>=0.5 and gt[i,j]==0:
                flag=1
        if flag==0:
            cnt = cnt+1
    return cnt/len(gt)



def one_class_accuracy(gt, pre, index):
    cnt=0.0
    for i in range(len(gt)):
        if pre[i,index]<0.5 and gt[i,index]==0:
            cnt=cnt+1
        elif pre[i,index]>=0.5 and gt[i,index]==1:
            cnt=cnt+1
    return cnt/len(gt)

CorrectPlot=[]
class EvaluateCallBack(Callback):
    def __init__(self, net, eval_dataset):
        super(EvaluateCallBack, self).__init__()
        self.net = net
        self.eval_dataset = eval_dataset

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        self.net.set_train(False)
        acc_list = []
        bas_list = []
        acc0 = []
        acc1 = []
        acc2 = []
        acc3 = []
        acc4 = []
        acc5 = []

        for item in self.eval_dataset.create_dict_iterator(num_epochs=1):
            image = item['image'].astype("float32")
            labels = item['label'].astype("float32")
            predict = self.net(image)
            for i in range(6):
                if i==0:
                    acc0.append(one_class_accuracy(labels, predict, i))
                if i==1:
                    acc1.append(one_class_accuracy(labels, predict, i))
                if i==2:
                    acc2.append(one_class_accuracy(labels, predict, i))
                if i==3:
                    acc3.append(one_class_accuracy(labels, predict, i))
                if i==4:
                    acc4.append(one_class_accuracy(labels, predict, i))
                if i==5:
                    acc5.append(one_class_accuracy(labels, predict, i))
            acc_list.append(count_accuracy(labels, predict))  

        acc = np.mean(acc_list)*100
        acc0_= np.mean(acc0)*100
        acc1_= np.mean(acc1)*100
        acc2_= np.mean(acc2)*100
        acc3_= np.mean(acc3)*100
        acc4_= np.mean(acc4)*100
        acc5_= np.mean(acc5)*100
        CorrectPlot.append(acc)
        print("------------------------------------------")
        print("Accuracy of scab: {}%".format(acc0_))
        print("Accuracy of healthy: {}%".format(acc1_))
        print("Accuracy of frog_eye_leaf_spot: {}%".format(acc2_))
        print("Accuracy of rust: {}%".format(acc3_))
        print("Accuracy of complex: {}%".format(acc4_))
        print("Accuracy of powdery_mildew: {}%".format(acc5_))
        print("epoch:{} eval_acc:{:.4f}%".format(cur_epoch_num,acc))
        print("------------------------------------------")
        save_checkpoint(net, "resnet101_epoch{}_acc{:.4f}.ckpt".format(cur_epoch_num,acc))


class LossforMuitilabel(nn.LossBase):
    def __init__(self, fun, reduction="mean"):
        super(LossforMuitilabel, self).__init__(reduction)
        self.fun = fun
    def construct(self, base, target):
        x0 = self.fun(base[:,0],target[:,0].astype("float32"))
        x1 = self.fun(base[:,1],target[:,1].astype("float32"))
        x2 = self.fun(base[:,2],target[:,2].astype("float32"))
        x3 = self.fun(base[:,3],target[:,3].astype("float32"))
        x4 = self.fun(base[:,4],target[:,4].astype("float32"))
        x5 = self.fun(base[:,5],target[:,5].astype("float32"))
        return (x0+x1+x2+x3+x4+x5)




# # 训练集路径
# train_path = '/data/users/user2/xjm/demo/plant_dataset/plant_dataset/train'
# # 验证集路径
# val_path = '/data/users/user2/xjm/demo/plant_dataset/plant_dataset/val'
# # 测试集路径
# test_path = '/data/users/user2/xjm/demo/plant_dataset/plant_dataset/test'

# # 读取train,val路径下的csv并合并，把val的图片移到train中
# train_csv = pd.read_csv(train_path + '/train_label.csv')
# val_csv = pd.read_csv(val_path + '/val_label.csv')
# train_csv = pd.concat([train_csv, val_csv], axis=0)
# train_csv.to_csv(train_path + '/train_label.csv', index=False)
# images = os.listdir(val_path + '/images')
# for image in images:
#     shutil.move(val_path + '/images/' + image, train_path + '/images/' + image)

datacsv_path_train='/data/users/user2/xjm/demo/plant_dataset/plant_dataset/train/train_label.csv'
dataimg_path_train='/data/users/user2/xjm/demo/plant_dataset/plant_dataset/train/images'
datacsv_path_test='/data/users/user2/xjm/demo/plant_dataset/plant_dataset/test/test_label.csv'
dataimg_path_test='/data/users/user2/xjm/demo/plant_dataset/plant_dataset/test/images'
print("Create dataset........")
dataset_train = create_dataset(KF_Dataset(datacsv_path_train,dataimg_path_train,spilt = "train"),
 batch_size=30, target='train', image_size=473)
dataset_val = create_dataset(KF_Dataset(datacsv_path_test,dataimg_path_test, spilt = "val"),
 batch_size=30, target='val', image_size=473)


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
dataset_sink_mode = True
dataset_size = dataset_train.get_dataset_size() #批次数据大小
print("dataset_size:",dataset_size)
net = resnet101(6)
loss_fn = LossforMuitilabel(nn.BCELoss(),reduction="mean") # 损失函数
time_cb = TimeMonitor(data_size=dataset_size)
loss_cb = LossMonitor(per_print_times=dataset_size)  
eval_cb = EvaluateCallBack(net, dataset_val)
optim = nn.Adam(net.trainable_params(),learning_rate=1e-3,weight_decay=0.0001)
model = Model(net, loss_fn, optim)
epochs = 100
print('Train epoch number = %d' % epochs)
print("-------------------------开始训练--------------------------------")
model.train(epochs, dataset_train, callbacks=[time_cb,loss_cb,eval_cb],dataset_sink_mode=dataset_sink_mode)
print("-------------------------结束训练------------------------------")

---
title: Pytorch命令记录
weight: 2
---
Pytorch命令记录
<!-- more -->
## 模型训练

### 随机种子设置
```python
#设置随机种子
import random
torch.backends.cudnn.deterministic = True#将cudnn框架中的随机数生成器设为确定性模式
torch.backends.cudnn.benchmark = False#关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### GPU多卡训练

```python
from torch.nn.parallel import DataParallel

device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
model = ResNet18().to(device)
model = DataParallel(model, device_ids=[0, 1]) # 额外加这一行
print(model)
```

### 预训练模型加载
`model.load_state_dict(torch.load('model.pth'))`
或者
`model = torch.load('model.pth') `
## 数据预处理
### 图像增强

```python
class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
```

```python
transform_train = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomResizedCrop(32,scale=(0.64,1.0),ratio=(1.0,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4823, 0.4465],[0.2023,0.1920,0.2210]), # 3通道彩色图像才需要正则化
    Cutout(n_holes=1, length=16)
])
transform_test = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4823, 0.4465],[0.2023,0.1920,0.2210])
])
```

## 数据集
### 划分数据集
`train_loader, valid_loader, test_loader = split_data_loader(train_data, test_data)`

```python
from torch.utils.data.sampler import SubsetRandomSampler
def split_data_loader(train_data, test_data, batch_size=128, num_workers=2, vaild_size=0.2):
    vaild_size = 0.2
    batch_size = 128
    num_workers = 2
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(vaild_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers)
    return train_loader, valid_loader, test_loader
```

## 模型训练

### 训练函数（周期更新学习率）

### 训练函数（周期更新学习率）

`train_list,valid_list, acc_list = train(model, train_loader,valid_loader, num_epochs=100, lr=0.01, wd=5e-4, devices=device, lr_period=30, lr_decay=0.5)`

`from tqdm import tqdm #进度条`

```python
# 训练函数参数分别是模型、训练集、验证集、训练轮数、学习率、权重衰减、设备、学习率调整周期、学习率衰减率
def train(model, train_loader,valid_loader, num_epochs, lr, wd, devices, lr_period, lr_decay): 
    train_list = [] # 记录训练集的loss
    valid_list = [] # 记录验证集的loss
    acc_list = [] # 记录验证集的准确率
    valid_loss_min = np.Inf # 记录最小的验证集loss

    criterion = nn.CrossEntropyLoss().to(device) # 使用交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=wd) # 使用SGD优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay) # 动态调整学习率
    for epoch in tqdm(range(1,num_epochs+1)):
        # 保存训练集和验证集的loss
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0
        #训练模式
        model.train()
        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(devices), label.to(devices)
            optimizer.zero_grad() # 梯度清零
            output = model(data).to(devices) # 前向传播
            loss = criterion(output, label.long()) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            train_loss += loss.item() * data.size(0) # 更新累计损失
        scheduler.step() # 更新学习率
        # 验证模式
        model.eval()
        for data, label in valid_loader:
            data, label = data.to(devices), label.to(devices)
            output = model(data).to(devices) # 前向传播
            loss = criterion(output, label.long()) # 计算损失
            valid_loss += loss.item() * data.size(0) # 更新累计损失
            _, pred = torch.max(output, 1) # 预测类别
            correct_tensor = pred.eq(label.data.view_as(pred)) # 判断预测类别与实际类别是否相等
            correct = np.squeeze(correct_tensor.cpu().numpy()) # 将tensor转为numpy
            total_sample += label.size(0) # 更新累计样本数
            right_sample += correct.sum().item() # 更新正确样本数
        
        # 计算平均损失
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        acc = right_sample / total_sample
        train_list.append(train_loss)
        valid_list.append(valid_loss)
        acc_list.append(acc)
        print("Epoch: {} 当前lr为：{}".format(epoch,optimizer.param_groups[0]['lr'])) # 显示当前学习率
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))# 显示训练集与验证集的损失函数 
        print("Epoch：", epoch, "valid_acc:", acc) # 显示验证集的准确率
        # 如果验证集的损失函数减少了，就保存模型
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), 'model.pth')
            valid_loss_min = valid_loss
    return model,train_list,valid_list, acc_list # 返回模型和训练集、验证集的损失函数纪录
```
### 训练函数（根据验证集更新学习率）
10次`loss`没下降则更新学习率`lr`

### wandb训练函数（周期更新学习率）

`train_list,valid_list,acc_list = train(model,'Project_Name',train_loader,valid_loader, num_epochs=100, lr=0.01, wd=0.0001, devices=device, lr_period=10, lr_decay=0.5)`

```python
# 训练函数参数分别是模型、wandb记录名称、训练集、验证集、训练轮数、学习率、权重衰减、设备、学习率调整周期、学习率衰减率
def train(model,log_name,train_loader,valid_loader, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 使用wandb跟踪训练过程,需登录，否则随机分配一个网址，登陆后获取密钥，在终端执行wandb login，输入密钥
    experiment = wandb.init(project=log_name, resume='allow', anonymous='must')
    train_list = [] # 记录训练集的loss
    valid_list = [] # 记录验证集的loss
    acc_list = [] # 记录验证集的准确率
    valid_loss_min = np.Inf # 记录最小的验证集loss

    criterion = nn.CrossEntropyLoss().to(device) # 使用交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=wd) # 使用SGD优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay) # 动态调整学习率
    for epoch in tqdm(range(1,num_epochs+1)):
        # 保存训练集和验证集的loss
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0
        #训练模式
        model.train()
        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(devices), label.to(devices)
            optimizer.zero_grad() # 梯度清零
            output = model(data).to(devices) # 前向传播
            loss = criterion(output, label.long()) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            train_loss += loss.item() * data.size(0) # 更新累计损失
        scheduler.step() # 更新学习率
        # 验证模式
        model.eval()
        for data, label in valid_loader:
            data, label = data.to(devices), label.to(devices)
            output = model(data).to(devices) # 前向传播
            loss = criterion(output, label.long()) # 计算损失
            valid_loss += loss.item() * data.size(0) # 更新累计损失
            _, pred = torch.max(output, 1) # 预测类别
            correct_tensor = pred.eq(label.data.view_as(pred)) # 判断预测类别与实际类别是否相等
            correct = np.squeeze(correct_tensor.cpu().numpy()) # 将tensor转为numpy
            total_sample += label.size(0) # 更新累计样本数
            right_sample += correct.sum().item() # 更新正确样本数

        # 计算平均损失
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        acc = right_sample / total_sample
        train_list.append(train_loss)
        valid_list.append(valid_loss)
        acc_list.append(acc)
        experiment.log({
            'valid_acc': acc,
            'train_loss': train_loss,
            'valid_loss':valid_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        # 如果验证集的损失函数减少了，就保存模型
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), 'model.pth')
            valid_loss_min = valid_loss
    return train_list,valid_list, acc_list # 返回模型和训练集、验证集的损失函数纪录
```

## 测试部分

### 测试函数

```python
def test(model,test_loader):
    total_sample = 0
    right_sample = 0
    model.eval()  # 验证模型
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)    
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        right_sample += correct_tensor.sum().item()
    acc = right_sample / total_sample
    return str(acc)+"%"
```
### 单张图片测试

```python
model.eval()
img = Image.open('/kaggle/working/train/100.png')
img = transforms.ToTensor()(img)
img = img.to(device)
label = model(img).argmax().item() #返回指定维度最大值的序号
class_name = {0:'frog', 1:'truck', 2:'deer', 3:'automobile', 4:'bird', 5:'horse', 6:'ship', 7:'cat', 8:'dog', 9:'airplane'}
print("预测值是：%s,真实值是：%s" % (class_name[label], img_list[100][1]))
```

## Matplotlib

### 一图绘制多条折线

```python
plt.title('train_loss & valid_loss & acc')
plt.plot(train_list, label='train_loss')
plt.plot(valid_list, label='valid_loss')
plt.plot(acc_list, label='acc')
plt.legend()
plt.show()
```

## NLP系列

### 词频统计

```python
# 导入BOW（词袋模型）
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])
```



## Gradio系列

### Gradio部署图片处理

```python
import gradio as gr

def image_classifier(inp):
    return inp

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="text")
demo.launch()
```

## 常用网络模型

### ResNet

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 7
model = ResNet18()
"""
ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
同时减小该卷积层的步长和填充大小
"""
# 处理 32x32 大小的3通道彩色图片
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉
model = model.to(device)
```

```python
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def ResNet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2],**kwargs)


def ResNet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3],**kwargs)


def ResNet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3],**kwargs)


def ResNet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3],**kwargs)


def ResNet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3],**kwargs)
```

## Q&A

### 将`numpy.ndarray`转为`PIL.Image.Image`
请注意，numpy数组的dtype必须为uint8，因为PIL只能处理8位无符号整数。如果您的numpy数组的dtype不是uint8，请使用numpy.astype()将其转换为uint8。
```python
import numpy as np  
from PIL import Image  
  
# 创建一个 numpy 数组  
arr = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)  
  
# 将 numpy 数组转换为 PIL 图像  
img = Image.fromarray(arr)  
  
# 显示图像  
img.show()
```
### Cloab自动断开

```javascript
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
```

### Wandb

```python
!pip install wandb
import wandb
wandb.login()
```

```python
# 使用wandb跟踪训练过程,需登录，否则随机分配一个网址，登陆后获取密钥，在终端执行wandb login，输入密钥
experiment = wandb.init(project="test", resume='allow', anonymous='must')
experiment.log({
    'train loss': loss_sum / len(train_loader),
    'valide loss': epoch,
})
```


---
title: 图神经网络DGL
title: Drain3
weight: 2
---
图神经网络学习记录
<!-- more -->
## 构建图结构

### 有向图

![](https://data.dgl.ai/asset/image/user_guide_graphch_1.png)
``` python
import dgl
import torch

# u->v 表示边， 边 0->1, 0->2, 0->3, 1->3
u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))

# 获取节点的ID
print(g.nodes())

# 获取边的对应端点
print(g.edges())

# 获取边的对应端点和边ID
print(g.edges(form='all'))

# 如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。
#g = dgl.graph((u, v), num_nodes=8)
```

### 无向图

```python
import dgl
import torch

# u->v 表示边
u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v)) # g为有向图

bg = dgl.to_bidirected(g)
```

### 异构图

定义为：节点或边的类型不同

>  如果是一个图，表示客户买电视，那这个图就有两种节点类型，客户和电视；而另一个图，表示人与人之间的关系，这个图里的节点都是人，是同一种类型。

## 图节点和边特征

关于 [`ndata`](https://docs.dgl.ai/en/latest/generated/dgl.DGLGraph.ndata.html#dgl.DGLGraph.ndata) 和 [`edata`](https://docs.dgl.ai/en/latest/generated/dgl.DGLGraph.edata.html#dgl.DGLGraph.edata) 接口的重要说明：

- 仅允许使用数值类型（如单精度浮点型、双精度浮点型和整型）的特征。这些特征可以是标量、向量或多维张量。
- 每个节点特征具有唯一名称，每个边特征也具有唯一名称。节点和边的特征可以具有相同的名称（如上述示例代码中的 `'x'` ）。
- 通过张量分配创建特征时，DGL会将特征赋给图中的每个节点和每条边。该张量的第一维必须与图中节点或边的数量一致。 不能将特征赋给图中节点或边的子集。
- 相同名称的特征必须具有相同的维度和数据类型。
- 特征张量使用”行优先”的原则，即每个行切片储存1个节点或1条边的特征（参考上述示例代码的第16和18行）。

```python
import dgl
import torch

g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0])) # 6个节点，4条边

g.ndata['x'] = torch.ones(g.num_nodes(), 3) # 长度为3的节点特征
g.edata['x'] = torch.ones(g.num_edges(), dtype=torch.int32) # 长度为4的边特征

#不同名称的特征可以具有不同形状
g.ndata['y'] = torch.randn(g.num_nodes(), 5) # 长度为5的节点特征

# 获取节点1的特征
g.ndata['x'][1]

# 获取边0和3的特征
g.edata['x'][torch.tensor([0, 3])]

weights = torch.tensor([0.1, 0.6, 0.9, 0.7])  # 每条边的权重
g.edata['w'] = weights  # 将其命名为 'w'
g
```

- trian_mask：一个布尔张量，指示节点是否在训练集中
- val_mask：一个布尔张量，指示节点是否在训练集中
- test_mask：一个布尔张量，指示节点是否在训练集中
- label：真实标签节点类别
- feat：节点的特征

## 从外部库创建图

### 从SciPy稀疏矩阵创建DGL图

``` python
import dgl
import torch as th
import scipy.sparse as sp

spmat = sp.rand(100, 100, density=0.05) # 5%非零项
g = dgl.from_scipy(spmat)
```

### NetworkX图创建DGL图

```python
import dgl
import torch as th
import networkx as nx

nx_g = nx.path_graph(5) # 一条链路0-1-2-3-4
g = dgl.from_networkx(nx_g) # 默认无向图
```

```python
# 有向图
nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
g = dgl.from_networkx(nxg)
```

## 数据集构建

``` python
import dgl
from dgl.data import DGLDataset
import torch

class DARPADataSet(DGLDataset):
    def __init__(self,nodes_data,triple_data):
        self.nodes_data = nodes_data
        self.triple_data = triple_data
        self.num_nodes = nodes_data.shape[0]
        super().__init__(name="Darpa DataSet")

    def __getitem__(self, idx):
        return self.graph

    # 返回图的数量
    def __len__(self):
        return 1
    
    # 处理函数
    def process(self):
        # 处理点节点的特征
        node_features = self.nodes_data["features"]
        node_label = self.nodes_data["label"]
        t_node_features = torch.tensor(node_features) # 转为tensor
        t_node_label = torch.tensor(node_label) # 转为tensor
        # 处理边节点的特征
        edge_src = torch.from_numpy(self.triple_data["id1"].to_numpy())
        edge_dst = torch.from_numpy(self.triple_data["id2"].to_numpy())
        edge_type = torch.from_numpy(self.triple_data["relation"].to_numpy())
        edge_timestamp = torch.from_numpy(self.triple_data["timestamp"].to_numpy())
        # 构造DGL图
        self.graph = dgl.graph((edge_src, edge_dst), num_nodes=self.num_nodes)
         # 添加点节点的两个特征
        self.graph.ndata["feat"] = t_node_features.float()
        self.graph.ndata["label"] = t_node_label.float()
        # 添加边节点的两个特征
        self.graph.edata["type"] = edge_type
        self.graph.edata["timestamp"] = edge_timestamp
        # 训练掩码
        num_train_nodes = int(self.num_nodes * 0.7)
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        correct_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        train_mask[:num_train_nodes] = True
        val_mask[num_train_nodes:] = True
        correct_mask[:] = False
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["correct_mask"] = correct_mask
```
## 参考资料

[入门图神经网络](https://zhuanlan.zhihu.com/p/617465069)

[DGL图神经网络下的节点分类（代码）](https://www.bilibili.com/video/BV18G411W7sd/?spm_id_from=333.337.search-card.all.click&vd_source=1fea5c6b53f4880bed7ab3ac56003802)


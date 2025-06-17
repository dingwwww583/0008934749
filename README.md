# 0008934749
 
## 功能说明

### 1. 数据预处理 (`data_preprocess.py`)
- 提供两种数据集加载方法：
  - `data_loader_FashionMNIST()`: 加载FashionMNIST数据集
  - `data_loader_MNIST()`: 加载标准MNIST数据集
- 数据集自动下载到`./data`目录
- 数据预处理：
  - 转换为Tensor格式
  - 自动归一化像素值到[0,1]范围
- 数据加载器配置：
  - 批大小：16
  - 丢弃最后不完整的批次
  - 训练集随机打乱
  - 返回训练集和测试集的DataLoader对象

### 2. 模型训练 (`train.py`)
- **模型架构**：
  ```python
  nn.Sequential(
      nn.Flatten(),
      nn.Linear(784, 256),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
  )

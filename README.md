# AEMLKit

## 依赖

- [Boost](http://www.boost.org/)
- [Shark](http://image.diku.dk/shark/)

## Demo 数据

断铅、落球和敲击的声发射信号，每个信号由能量、持续时间和振铃计数三个参数值构成的特征向量表示。

- [`data_train.csv`](data_train.csv): 训练数据，第 1~40 行、41~80 行和 81~120 行分别是 40 组断铅、落球、敲击信号；
- [`data_validate.csv`](data_validate.csv): 验证数据，第 1~10 行、11~20 行和 21~30 行分别是 10 组断铅、落球、敲击信号；
- [`data_test.csv`](data_test.csv): 测试数据，第 1~10 行、11~20 行和 21~30 行分别是 10 组断铅、落球、敲击信号；

## Demo 说明

### 1. 训练数据格式

`TrainingData` 结构体代表一个训练数据，包含代表特征向量的 `double` 数组以及代表该训练数据所属类别的无符号整形（0, 1, 2...代表不同类别）。

```
struct TrainingData {
    double *input;
    unsigned int label;
};
```

### 2. 构造并初始化神经网络

`FeedForwardNet` 类代表神经网络，实例化时需要的参数有：

```
size_t numInputs;           // 输入节点数（即特征向量维数）
size_t numHiddenLayers;     // 隐藏层数
size_t numClasses;          // 输出节点数（即类别数）
size_t *hiddenLayerSizes;   // 各隐藏层神经元数
```

`void FeedForwardNet::initFFNet()` 完成神经网络的初始化

### 3. 训练

```
void FeedForwardNet::train(TrainingData *dataset, size_t count, unsigned int epochs)
```

参数分别为训练数据数组、训练数据个数和训练迭代次数。

### 4. 验证

```
double FeedForwardNet::validate(ValidatingData *dataset, size_t count)
```

可传入一组与训练数据结构相同的验证数据（`typedef TrainingData ValidatingData;`），返回神经网络对该数据集的错误分类率。

### 5. 预测

```
PredictResult FeedForwardNet::predict(double *input)
```

接收一个特征向量数组，返回神经网络对该样本的预测分类结果，返回数据格式为：

```
struct PredictResult {
    unsigned int predictedLabel;    // 预测输出（类别代号: 0, 1, ...）
    double *probabilities;          // 输入分别属于各类别的概率
};
```
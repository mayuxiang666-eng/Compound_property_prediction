# 方案 3：变长多元曲线用 GAK 支持向量回归，再看拿掉一个通道后的误差

这是一套完整方案。只按本文件做。不要和方案 1、方案 2 串起来。

tslearn 对不等长序列、标量输出，列出的回归模型就是核为 GAK 的 TimeSeriesSVR。它接受每车一个 `(T_i, d)`，`T_i` 可以不同。本方案用它拟合终胶门尼。一个通道是否有关，不看相关系数，看把这个通道在车与车之间打乱之后，留出批次的均方误差是否上升。

禁止用这些代替：Pearson、Spearman、Kendall、互信息、最大信息系数、catch22、路径签名、普通 RBF 支持向量回归、把曲线补齐后再做欧氏距离。不要改用 `kernel="rbf"`。

## 模型从哪来

GAK 与不等长回归。论文：Marco Cuturi. Fast global alignment kernels. Proceedings of the 28th International Conference on Machine Learning, 2011。

代码：https://github.com/tslearn-team/tslearn

不等长回归的说明：https://github.com/tslearn-team/tslearn/blob/main/docs/variablelength.rst

安装：`pip install tslearn`

```python
from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset
```

## 输入

一行是一车母胶。按终胶批次把车分成训练集和留出集。几车共用一个终胶门尼时，这些车必须进同一个集合，不要按车随机切。

- 每车一个数组，形状 `(T_i, d)`。列顺序固定为温度、功率、压栓。有转速就加在后面。各车 `d` 相同，`T_i` 可以不同。
- 停留时间不要放进这个数组。GAK 会把沿时间重复的常数和曲线长度缠在一起。
- 终胶门尼 `y` 是一维 `float`。

有 NaN 或 Inf 就停止。`to_time_series_dataset` 产生的 NaN 是短序列的填充，不要填成 0。

## 链路

1. 只用训练集拟合。

```python
X = to_time_series_dataset(curves)
model = TimeSeriesSVR(C=1.0, kernel="gak")
model.fit(X, y)
```

`C` 固定为 1。不要网格搜索。

2. 在留出集上计算均方误差，记为 `mse_full`。

3. 对每一个通道单独做一次：在留出集里打乱这一列在各车之间的归属，每条曲线的长度仍跟原车走，其他通道不动。用同一种子 `0`，每个通道只打乱一次。再用已经拟合好的模型预测，得到 `mse_without`。

4. 该通道的差是 `mse_without - mse_full`。差不大于 0，就说明打乱它没有让留出误差变大，这一通道不保留。

5. 停留时间不进入第 3 步。本方案不报告停留时间和门尼的相关。要看停留时间，用方案 1。

## 输出

一张表。一行一个通道。

- 通道名
- `mse_full`
- `mse_without`
- 两者的差

相关数就是这个差。差大于 0 才保留该通道。不要把这个差写成某一个时刻的相关，也不要写成因果。

# 方案 2：变长曲线用全局对齐核，再计算与终胶门尼的 HSIC

这是一套完整方案。只按本文件做。不要和方案 1、方案 3 串起来。

同一车里的温度、功率、压栓长度相同，不同车的长度可以不同。Cuturi 2011 年的全局对齐核就是为这种长度不同、维数相同的序列定义的，不需要先排成同一个长度。HSIC 用来判断这个核和终胶门尼之间有没有依赖。

禁止用这些代替：Pearson、Spearman、Kendall、互信息、最大信息系数、catch22、路径签名、把曲线插值或补齐后再算欧氏距离。DTW 距离不是正定核，不能拿来替换这里的 GAK。

## 模型从哪来

全局对齐核。论文：Marco Cuturi. Fast global alignment kernels. Proceedings of the 28th International Conference on Machine Learning, 2011。

代码在 tslearn。仓库：https://github.com/tslearn-team/tslearn

安装：`pip install tslearn`

```python
from tslearn.metrics import cdist_gak, sigma_gak
from tslearn.utils import to_time_series_dataset
```

两条序列长度可以不同，维数必须相同。`gak` 的值在 0 到 1 之间。

HSIC。论文：Arthur Gretton, Olivier Bousquet, Alex Smola, Bernhard Schölkopf. Measuring statistical dependence with Hilbert-Schmidt norms. Algorithmic Learning Theory, 2005。高斯核是特征核时，总体 HSIC 为 0 当且仅当独立。这里曲线之间用 GAK，门尼之间用高斯核。

## 输入

一行是一车母胶。只用训练批次。几车共用一个终胶门尼时，仍然一车一行，不要平均。

- 每车一个数组，形状 `(T_i, d)`。列顺序固定为温度、功率、压栓。有转速就加在后面。各车 `d` 相同，`T_i` 可以不同。
- 单通道检验时，每车是一条一维序列，长度就是该车的 `T_i`。
- 停留时间是形状 `(n,)` 的一个数。不要放进曲线，不要做 GAK。
- 终胶门尼 `y` 是形状 `(n,)` 的一维 `float`。

有 NaN 或 Inf 就停止。`to_time_series_dataset` 用来把不等长序列放进一个数组，短序列后面是 NaN。不要把这些 NaN 填成 0。

## 链路

1. 带宽用训练集估计，不要手写一个常数。

```python
X = to_time_series_dataset(curves)
sigma = float(sigma_gak(X))
K = cdist_gak(X, sigma=sigma)
```

多元曲线做一次，得到整车曲线和门尼的关系。温度、功率、压栓再各做一次，每次只放该通道。

2. 门尼的高斯核。`gamma` 用中位数：`gamma = 1 / median((y_i - y_j)^2)`，中位数只在 `i < j` 的差值上计算。若中位数为 0，就停止，不要改成一个小数。

```python
L = np.exp(-gamma * (y[:, None] - y[None, :]) ** 2)
```

3. 两个核都中心化。`H = I - 1/n`，`Kc = H @ K @ H`，`Lc` 同样。样本 HSIC 是

```python
hsic = np.sum(Kc * Lc) / (n - 1) ** 2
```

4. 把 `y` 随机打乱 1000 次，`random_state` 固定为 0，每次重算 `Lc` 和 HSIC，`K` 不动。p 值是打乱后 HSIC 大于等于实际 HSIC 的比例。`pvalue < 0.05` 才保留。阈值不要改。

5. 停留时间不进入 GAK。用 `xicorpy.compute_conditional_dependence(y, dwell)` 单独算它和门尼的系数。这一行不要写成“在曲线给定之后的条件相关”，本方案没有做这一步。

## 输出

一张表。

- 一行是多元曲线，一行是温度，一行是功率，一行是压栓，有转速再加一行，最后一行是停留时间。
- 曲线行写 HSIC 和 p 值。
- 停留时间行写条件依赖系数，p 值留空。

相关数是 HSIC 或停留时间那一行的条件依赖系数。不要把 HSIC 读成某一个时刻的相关，也不要写成因果。

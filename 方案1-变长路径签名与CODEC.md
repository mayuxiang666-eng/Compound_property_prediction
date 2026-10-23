# 方案 1：变长多元曲线做成路径签名，再用条件依赖系数筛选

这是一套完整方案。只按本文件做。不要和方案 2、方案 3 串起来。

这条数据是一车一条多元曲线，温度、功率、压栓在同一车里一起采样，车和车的长度可以不同，标签是一个终胶门尼。Fermanian 2022 年的签名线性模型就是为“向量值函数对应一个实数输出”写的，签名不要求各车点数相同。条件依赖系数用来回答签名的哪一项、以及停留时间，在其他项给定之后还和门尼有关。签名本身不给出这个条件相关，条件依赖系数也不能直接吃一条变长曲线，所以两步都要做，不能只留一步。

禁止用这些代替：Pearson、Spearman、Kendall、互信息、最大信息系数、catch22、DTW、GAK、把曲线插值或补齐成同一长度。不要调用 https://github.com/afermanian/signature-regression 里的实验脚本，那个仓库是论文的模拟实验，不是本方案的入口。

## 模型从哪来

路径签名。论文：Adeline Fermanian. Functional linear regression with truncated signatures. Journal of Multivariate Analysis, 2022, 192: 105031. DOI: 10.1016/j.jmva.2022.105031。预印本 arXiv:2006.08442。

计算用 iisignature。仓库：https://github.com/bottler/iisignature

安装：`pip install iisignature`

```python
import iisignature
```

安装失败就停止。不要改用 catch22 或别的特征库。

一条路径是形状 `(T, d)` 的数组，`T` 是这一车的点数，各车可以不同。`iisignature.sig(path, m)` 返回这一车的签名，长度由 `iisignature.siglength(d, m)` 决定，各车这个长度相同。

条件依赖系数。论文：Mona Azadkia, Sourav Chatterjee. A simple measure of conditional dependence. The Annals of Statistics, 2021, 49(6): 3070–3102. DOI: 10.1214/21-AOS2073。

仓库：https://github.com/swarnakumar/xicorpy

安装：`pip install xicorpy`

```python
import xicorpy
```

给定已选特征，该系数为 0 当且仅当终胶门尼与当前特征条件独立，为 1 当且仅当终胶门尼是该特征的函数。

## 输入

一行是一车母胶。只用训练批次。几车共用一个终胶门尼时，仍然一车一行，不要平均。

- 每车一个数组，形状 `(T_i, d)`。列的顺序固定为温度、功率、压栓。曲线里如果有转速，加在这三列后面，所有车的 `d` 必须相同。`T_i` 可以不同。
- 在这些列之后再加一列采样序号，从 0 到 `T_i - 1`。不要把序号除以 `T_i`，否则每车的时长会被抹平。
- 停留时间是形状 `(n,)` 的一个数。不要放进路径。
- 终胶门尼 `y` 是形状 `(n,)` 的一维 `float`。

某一车里有 NaN 或 Inf 就停止，不要插值。

## 链路

1. 截断阶数 `m` 固定为 2。`d` 含采样序号。不要改成 3 或 4。

```python
sig = iisignature.sig(path, 2)
```

每车得到一条同样长度的向量。列名写成 `签名_000` 这种三位下标，顺序与 `sig` 的返回顺序一致。

2. 拼成 DataFrame，形状 `(n, p)`，再加一列 `停留时间`。按训练集标准化。任一列在全部车上取值相同，就删掉该列。

3. 用 FOCI 的规则逐列选。还没有已选列时，`given` 传 `None`。已有已选列时，`given` 是这些列拼成的数组。

```python
value = xicorpy.compute_conditional_dependence(y, X[name].to_numpy(), given)
```

每一轮选系数最大的列。系数不大于 0 就停止。

不要用 `select_features_using_foci(..., get_conditional_dependency=True)` 返回的那串数。那个数是已选全体和门尼的系数，不是当前这一列的条件依赖系数。

## 输出

一张表。一行一个被选入的签名项或停留时间。

- 列名
- 选入顺序
- 选入时的条件依赖系数
- 使用的 `d` 和 `m`

相关数只采用条件依赖系数。签名下标不是曲线上的某一个时刻。不要把表里的数写成因果。

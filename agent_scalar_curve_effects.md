# Agent 说明：多变量曲线 → 一个标量结果

另一个 agent 只读这一份。不要再搜方法。不要改下面四个库的算法，只按这里的入口、输入形状和参数来跑。

## 任务

每个样本是一条多变量过程曲线，结果 `y` 是一个标量（门尼或能耗）。要回答：

1. 单个特征如何影响 `y`（方向、幅度、发生在曲线的哪一段）。
2. 哪几个特征合在一起的影响，超过它们各自单独的影响。

## 先看哪段数据

- 曲线还在，且能按工步对齐到同一时间网格：用第 1 节 FDboost。
- 手里已经是工步标量或窗口标量：按 2 → 3 → 4 的顺序。第 2 节只筛单特征。第 3、4 节才写组合。

仓库里已有、不要重写：

- `PalletMnyV1/pallet_mny/hsic_lasso.py`：Yamada 2014 的简化版，用的是 `sklearn.linear_model.Lasso(alpha=...)`。它不是官方包，没有块估计，也没有 `covars`。新分析调用 `pyHSICLasso`。
- `PalletMnyV1/pallet_mny/tuples.py`：订单分组下 Ridge 的 R² 增量。评价协议照这个分组方式，模型换成下面的库。
- `MnyCorrV2/MODEL.md`：前向验证、近期均值基线、订单内置换。最终数字按那一节，不按各库自带的随机交叉验证。

## 数据合同

记 `n` 为样本数，`p` 为特征数，`G` 为对齐后的时间点数。

| 对象 | 形状 | 约定 |
|---|---|---|
| 一条曲线的一个通道 | `(n, G)` | 第 i 行是第 i 个样本。所有样本共用同一网格 `s`，长度 G。缺失值先处理掉，FDboost 的函数型协变量不接受 NA。 |
| 工步标量表 `X` | `(n, p)` | 一行一个样本。列名保留，后面三个方法都要传名字。 |
| 标量结果 `y` | `(n,)` | 一维。门尼或能耗，不要把两条目标拼成二维一起喂给 HSIC。 |
| 分组 | 长度 n | `OrderID`。同一订单的样本不能拆进训练和测试。 |
| 时间顺序 | 长度 n | 订单投产时间。前向验证只用测试块之前的订单训练。 |
| 混杂 | `(n, c)` | 配方、机台等。HSIC 用 `covars`。不要把它们当过程特征去解释「怎么影响门尼」。 |

曲线长度不齐时，先按工步边界对齐到同一网格，再进 FDboost。不对齐就不要调用 `bsignal`。

## 固定评价（四个方法共用）

各库内部的随机验证只用于它们自己选复杂度。写进结论的数字必须重算：

1. 切分：按订单时间前向验证，或 `GroupKFold` 按 `OrderID`。对照基线是全局均值和近期订单均值。
2. 单特征有没有依赖：在订单内打乱 `y`，看统计量是否超过空分布。多个特征用 Benjamini-Hochberg，假发现率 0.10。
3. 方向和幅度：报原单位。FDboost 报 `β(t)`，EBM 报 `f_j`，glinternet 报 `β_j`。区间来自分组 bootstrap。区间盖住 0 的时段不解释。
4. 组合是否更大。同一套切分上计算：

```text
ΔR² = R²(主效应 + 该组合) − R²(仅主效应)
```

再在订单内打乱 `y`，得到 `ΔR²` 的空分布。单侧 p 和幅度一起过线，才写「组合更强」。只报全模型 R² 会把主效应算进组合。

5. 稳定性：同一特征或同一对要在多数时间块里再次被选中。

预测 R² 单独报，并和近期均值比。弱但稳定的效应可以解释均值移动，同时预测不了单车。设定值是能改的量；功率、温度、能耗是过程结果。写「影响」时标明是条件关联，还是控制混杂后的效应。后者沿用现有 DML，不要把 EBM 形状图写成因果。

不要用：IMV-LSTM（它预测的是目标序列的下一步，不是批次标量）、在打不赢近期均值的模型上读 SHAP 或 Friedman H。

---

## 1. FDboost：曲线上的 β(t)，以及两条曲线的交互

### 来源

- 方法：Brockhaus, Rügamer, Greven. Boosting Functional Regression Models with FDboost. Journal of Statistical Software, 2020. https://doi.org/10.18637/jss.v094.i10
- 代码：https://github.com/boost-R/FDboost ，CRAN 包 `FDboost`（依赖 `mboost`）
- 函数帮助：`?FDboost`，`?bsignal`，`?validateFDboost`，`?boost_control`

标量模型：

```text
y = Σ_j ∫ x_j(t) · β_j(t) dt
```

`β_j(t)` 是特征 j 在时刻 t 的影响。两条曲线的组合是 `bsignal %X% bsignal` 的交互曲面。

### 安装与最小调用

```r
install.packages("FDboost")
library(FDboost)

# dat$X1, dat$X2: 数值矩阵，n 行、G 列，一行一条曲线
# dat$s: 长度 G 的网格，所有曲线共用
# dat$y: 长度 n 的标量
# 协变量按列中心化。官方示例用 scale(..., scale = FALSE)
dat$X1 <- scale(dat$X1, scale = FALSE)
dat$X2 <- scale(dat$X2, scale = FALSE)

mod <- FDboost(
  y ~ 1 + bsignal(X1, s, knots = 10, df = 4) +
        bsignal(X2, s, knots = 10, df = 4),
  timeformula = NULL,
  data = dat,
  family = Gaussian(),
  control = boost_control(mstop = 100, nu = 0.1)
)

# 两条曲线的交互。df 必须和主效应相同，否则更灵活的项会被优先选中
mod_int <- FDboost(
  y ~ 1 + bsignal(X1, s, knots = 10, df = 4) %X%
           bsignal(X2, s, knots = 10, df = 4),
  timeformula = NULL,
  data = dat,
  control = boost_control(mstop = 100, nu = 0.1)
)

cv <- validateFDboost(mod, grid = 1:500)
mstop(mod) <- mstop(cv)
plot(mod)
coef(mod)
```

`timeformula = NULL` 表示 y 是标量。写成 `~ bbs(t)` 会把 y 当成函数型响应，本任务不要这样写。

### 要改的参数

- `boost_control(mstop)`：默认 100，帮助页写明这个默认值通常不够。先放到一个偏大的网格（例如 `1:500` 或 `1:1000`），再用 `validateFDboost` 选。最终用选中的 `mstop`，不要把网格上界本身当结果。
- `boost_control(nu)`：步长。保持 `0.1`。要更稳就降到 `0.05`，同时把 `mstop` 网格加长。不要为了少迭代把 `nu` 调到 0.5 以上。
- `bsignal(..., df)`：单个基学习器的复杂度，默认 4。所有 `bsignal` 用同一个 `df`。特征很多、样本不多时降到 3。曲线很弯、且 `validateFDboost` 把 `mstop` 顶到网格上界时，升到 6，并检查 `extract(mod, "df")`。
- `bsignal(..., knots)`：默认 10。网格 G 很密（远大于 100）且 β(t) 明显只有少数拐点时保持 10。工步内形状很尖、10 个结不够时升到 20 或 40。`knots` 是内部结点数或结点位置向量，不是时间点数。
- `bsignal(..., differences)`：差分阶，默认 1。β(t) 要更光滑时改为 2。
- `family`：均值回归用默认 `Gaussian()`。y 有个别极端化验值、只关心中位数时改为 `QuantReg()`。

### 不要改

- 不要给不同通道设不同的 `df`。帮助页写明：df 更大的基学习器更容易被选中，比较会偏。
- 不要在 `bsignal` 里留 NA。
- 交互项 `%X%` 的预测在新数据上依赖较新的 FDboost。CRAN 当前版本可用；旧版本对 `bsignal %X% bsignal` 的 `predict` 有过错误，不要换未更新的 fork。

### 读结果

`plot(mod)` 画出 β(t)。纵轴离开 0 的时段就是该通道在起作用的时段。`coef(mod)` 是系数。交互模型看交互曲面，不把曲面再平均成一个数。

稳定性用拟合对象上的 `stabsel()`（包内有该方法）。改选入频率或每子样本选取个数之前先读 `?stabsel`。被多数子样本选中、且 bootstrap 区间不盖住 0 的时段，才写入结论。

---

## 2. pyHSICLasso：非线性、去冗余的单特征筛选

### 来源

- 方法：Yamada et al. Neural Computation, 2014. https://doi.org/10.1162/NECO_a_00537
- 大样本块估计：Climente-González et al. Bioinformatics, 2019. https://doi.org/10.1093/bioinformatics/btz333
- 代码：https://github.com/riken-aip/pyHSICLasso
- 安装：`pip install pyHSICLasso`
- 实现：`pyHSICLasso/api.py` 里的 `HSICLasso.regression`

只负责单特征短名单。不要把两个特征的乘积列塞进来充作组合效应；组合放到第 3、4 节。

### 输入形状（这里最容易用错）

`input(X, y)` 接受的 `X` 是 `(n_samples, n_features)`，`y` 是一维 `(n_samples,)`。类内部会把 `X` 转置成 `(n_features, n_samples)`。`y` 如果是二维，`_input_data_ndarray` 直接报错。

`covars` 的行数必须等于样本数 `n`。源码检查的是转置之后的 `X_in.shape[1] == covars.shape[0]`。

`B` 必须整除 `n`。不能整除时会 `RuntimeWarning`，并把块数截断。

### 最小调用

```python
import numpy as np
from pyHSICLasso import HSICLasso

hsic = HSICLasso()
# X: (n, p), y: (n,), names: 长度 p 的列名
hsic.input(X, y, featname=list(names))
hsic.regression(
    num_feat=10,
    B=0,
    M=1,
    discrete_x=False,
    max_neighbors=5,
    n_jobs=1,
    covars=np.asarray(covars, dtype=float),  # (n, c)；没有混杂就不要传，用默认空数组
    covars_kernel="Gaussian",
)
selected = hsic.get_features()       # 列名，按入选顺序
idx = hsic.get_index()                # 列下标，0-based
scores = hsic.get_index_score()       # 与 idx 对齐
hsic.dump()
```

没有混杂时删掉 `covars=` 这个参数，使用默认 `np.array([])`。传空矩阵也可以，源码只在 `covars.size` 为真时校正。

### 要改的参数

签名默认值来自 `regression(self, num_feat=5, B=20, M=3, discrete_x=False, max_neighbors=10, n_jobs=-1, covars=np.array([]), covars_kernel="Gaussian")`。

- `num_feat`：要留下的特征个数，默认 5。这是停止个数，不是 Lasso 的 `alpha`。短名单给后面的 EBM / glinternet 用时设 10 到 20。设得比 `p` 大没有意义。仓库里手写版的 `lam=0.01` 不要搬过来，官方接口没有这个参数。
- `B`：块大小，默认 20。`n <= 1000` 时设 `B=0` 且 `M=1`，走完整 HSIC。`n > 1000` 时选一个能整除 `n` 的 `B`（文档要求 `B << n`），否则会警告并截断块数。`B` 越小越省内存，估计越噪。
- `M`：块估计的重排次数，默认 3。只在 `B` 不为 0 时有用。结果在不同 `M` 间抖动时升到 5。`B=0` 时保持 `M=1`。
- `discrete_x`：默认 `False`，所有特征用高斯核。过程量是连续的，保持 `False`。这个开关是全局的，不能一列连续、一列离散。类别列先拿出去，放进 `covars`（`covars_kernel="Delta"`）或单独做一次筛选。
- `covars_kernel`：连续混杂用 `"Gaussian"`，类别混杂用 `"Delta"`。混杂里两种都有时，先把类别做成 0/1 列再全部用 `"Gaussian"`，不要在同一次调用里混两种核。
- `max_neighbors`：`dump()` 里展示的相关特征个数，默认 10。不改变谁被选中。
- `n_jobs`：默认 `-1`。要可复现的逐位比较时设 `1`。

### 读结果

`get_features()` 的顺序就是入选顺序，越靠前依赖越强，且后面的特征会扣掉与前面特征的冗余。分数是相对值。入选不等于「影响显著」：短名单还要用上面的订单内置换再滤一遍。

---

## 3. Explainable Boosting Machine：主效应曲线和配对曲面

### 来源

- 方法：Lou, Caruana, Gehrke, Hooker. Accurate Intelligible Models with Pairwise Interactions. KDD, 2013.
- 软件：Nori et al. InterpretML. https://arxiv.org/abs/1909.09223
- 代码：https://github.com/interpretml/interpret
- 安装：`pip install interpret`
- 类：`interpret.glassbox.ExplainableBoostingRegressor`
- 参数页：https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html

模型：

```text
y = Σ_j f_j(x_j) + Σ_(j,k) f_jk(x_j, x_k)
```

`f_j` 是一条曲线，`f_jk` 是一张曲面。三元项可以写进 `interactions`，但先不要开。

### 最小调用

```python
from interpret.glassbox import ExplainableBoostingRegressor

# 只解释 HSIC 短名单上的配对。元组是 X 的列下标，0-based。
pair_list = [(0, 1), (0, 4), (2, 3)]

ebm = ExplainableBoostingRegressor(
    feature_names=list(names),
    interactions=pair_list,       # 或一个小整数，见下
    outer_bags=8,
    learning_rate=0.04,
    max_rounds=5000,
    early_stopping_rounds=100,
    min_samples_leaf=4,
    validation_size=0.15,         # 只给早停用；最终指标不要用这次随机划分
    random_state=42,
    n_jobs=1,
)
ebm.fit(X, y)
ebm.predict(X)
# 每个可加项一列分数，列顺序与 ebm.term_features_ 相同
term_scores = ebm.eval_terms(X)
```

全局图：`ebm.explain_global()`。项的组成：`ebm.term_features_`（特征下标元组）、`ebm.term_names_`、`ebm.term_scores_`。

### 要改的参数

默认值来自当前文档中的 `ExplainableBoostingRegressor`。

- `interactions`：默认 `"5x"`，即 `5 * p` 个自动配对。`p = 74` 时就是 370 对，不可解释。改成整数 `10`（只要 10 对），或改成上面的元组列表（只检验事先指定的对）。三元写成三元组，例如 `(0, 1, 2)`。指定了高阶项、又要同时提升主效应时，按文档把 `exclude="mains"` 且 `max_bins == max_interaction_bins`。本任务不要走这条，主效应和配对分两阶段即可。
- `learning_rate`：默认 `0.04`。早停很早、曲线还在抖时降到 `0.01`，并把 `max_rounds` 加大。不要只加大 `learning_rate` 来换速度。
- `max_rounds`：默认 `50000`。样本只有几百时降到 `2000`–`5000`，靠 `early_stopping_rounds` 停。
- `early_stopping_rounds`：默认 `100`。设 `0` 会关停早停，一直跑满 `max_rounds`。
- `outer_bags`：默认 `14`，用来做误差带和光滑。样本少时降到 `4` 或 `8`。误差带要更稳就升回 `14`。
- `min_samples_leaf`：默认 `4`。工步特征里有稀疏档位、叶节点太碎时升到 `10`。
- `max_bins`：主效应分箱上限，默认 `1024`。连续特征的有效分箱由数据决定。形状被噪声打成锯齿时降到 `64` 或 `256`。
- `max_interaction_bins`：配对分箱上限，默认 `64`。配对曲面太粗时升到 `128`；样本不足以填满格子时降到 `32`。
- `objective`：默认 `"rmse"`。y 是门尼或能耗时保持 `"rmse"`。极端值主导损失时改为 `"pseudo_huber:delta=1.0"`。
- `random_state`：默认 `42`。比较运行之间的差异时固定它。`None` 不可复现。
- `greedy_ratio`：默认 `10.0`。要复现 2024-02 之前那种纯轮询、少改形状的旧行为，设 `0`。

### 不要把内部验证当结论

`validation_size=0.15` 是从样本里随机抽验证集做早停。门尼随订单时间漂移，这次划分会泄漏。最终 R² 和 `ΔR²` 必须按「固定评价」重算。

若要让早停本身也遵守时间顺序，给 `fit` 传 `bags`：形状 `(n_samples, outer_bags)`，`+1` 训练，`-1` 验证，`0` 不参与。验证行只能是更晚的订单。

### 读结果

`term_features_` 里长度为 1 的元组是主效应，长度为 2 的是配对。`term_scores_` 是该特征取值（或取值组合）对应的加成分，单位与 `y` 相同（`objective="rmse"` 且链接为恒等时）。配对项要保留，还得通过 `ΔR²` 的置换检验，不能只看曲面图很好看。

---

## 4. glinternet：带强层次约束的线性配对

### 来源

- 方法：Lim, Hastie. Journal of Computational and Graphical Statistics, 2015. https://doi.org/10.1080/10618600.2014.938812
- 代码：CRAN 包 `glinternet` 1.0.13。镜像 https://github.com/cran/glinternet
- 帮助：`?glinternet`，`?glinternet.cv`，`?coef.glinternet`

模型：

```text
y = μ + Σ_j β_j x_j + Σ_(j<k) β_jk x_j x_k
若 β_jk ≠ 0，则 β_j ≠ 0 且 β_k ≠ 0
```

连续过程特征用它。EBM 给非线性形状，glinternet 给短的线性组合名单。两处都出现的对，才写入结论。

### 最小调用

```r
install.packages("glinternet")
library(glinternet)

# X: n x p 数值矩阵。连续列的 numLevels 全是 1。
# 下标从 1 开始，不是 Python 的 0。
p <- ncol(X)
numLevels <- rep(1L, p)

# 只允许 HSIC 短名单内部的配对。下面是 1-based 示例。
pairs <- matrix(c(1, 2,
                   1, 5,
                   3, 4), ncol = 2, byrow = TRUE)

fit <- glinternet(
  X = X,
  Y = y,
  numLevels = numLevels,
  nLambda = 50,
  lambdaMinRatio = 0.01,
  interactionPairs = pairs,
  family = "gaussian",
  numCores = 1
)

# lambdaIndex 从 1 开始。越大，lambda 越小，项越多。
# 用前向验证挑一个 index，不要默认取路径尽头。
coefs <- coef(fit, lambdaIndex = 20)
coefs$mainEffects$cont
coefs$mainEffectsCoef$cont
coefs$interactions$contcont
coefs$interactionsCoef$contcont
```

`glinternet.cv(..., nFolds = 10)` 的折是随机的，只能用来看路径大概停在哪。报告用的 `lambdaIndex` 用前向验证或按 `OrderID` 的分组误差来选。选中之后用 `predict(fit, Xnew, lambda = fit$lambda[index])`。

### 要改的参数

签名：`glinternet(X, Y, numLevels, lambda = NULL, nLambda = 50, lambdaMinRatio = 0.01, interactionCandidates = NULL, interactionPairs = NULL, screenLimit = NULL, numToFind = NULL, family = c("gaussian", "binomial"), tol = 1e-05, maxIter = 5000, verbose = FALSE, numCores = 1)`。

- `numLevels`：每个变量的水平数，长度必须是 `p`。连续特征全部写 `1`。真正的类别写成水平个数，且该列编码为 `0, 1, ..., L-1`。过程曲线的工步标量是连续的，用全 1。
- `interactionPairs`：`m x 2` 矩阵，1-based 列号。有 HSIC 短名单时传它，搜索空间就只剩这些对。
- `interactionCandidates`：一条下标向量。只搜索「这些列 × 其余所有列」。与 `interactionPairs` 不要同时乱设；有明确的对就只用 `interactionPairs`。
- `screenLimit`：每个 lambda 上只和当前最强的这么多个主效应做交互。`p` 很大又没传 `interactionPairs` 时设 `20`。已经用 `interactionPairs` 限制过就不需要。
- `nLambda`：路径长度，默认 `50`。路径太稀、相邻 lambda 的活跃集跳得很厉害时升到 `80`。
- `lambdaMinRatio`：最小 lambda 相对最大 lambda 的比例，默认 `0.01`。路径尽头项仍然太少时降到 `0.001`。样本少、尽头明显过拟合时升到 `0.05`。
- `numToFind`：找到这么多个交互对就停。想要短名单时设 `10`。默认 `NULL` 会把整条路径算完。
- `family`：连续 y 用 `"gaussian"`。不要用 `"binomial"`。
- `maxIter`：默认 `5000`。`verbose = TRUE` 时看到某一 lambda 不收敛，再升到 `10000`。
- `tol`：默认 `1e-5`。不要为了快把它放到 `1e-2`。

`coef()` 的 `lambdaIndex` 是路径上的位置，不是 lambda 的数值。索引变大，惩罚变小，主效应和交互都变多。

### 读结果

`interactions$contcont` 是两列连续变量的 1-based 下标，`interactionsCoef$contcont` 是对应的 `β_jk`。层次约束由算法保证：出现的交互对，两个主效应也在 `mainEffects$cont` 里。系数已按原始量纲还原（拟合时内部做过标准化）。

---

## 推荐执行顺序

1. 曲线能对齐：跑 FDboost，记下 β(t) 离开 0 的时段。
2. 工步标量：`pyHSICLasso`，`B=0, M=1`（n≤1000）或能整除 n 的 `B`，`num_feat=10` 到 `20`，混杂放 `covars`。
3. 对短名单跑 EBM，`interactions` 用短名单内的元组或一个 ≤10 的整数。
4. 对同一短名单跑 glinternet，`interactionPairs` 用 1-based 下标，`numLevels` 全 1。
5. 用「固定评价」算每对的 `ΔR²` 和订单内置换 p。EBM 和 glinternet 都点到的对，优先写结论。
6. 只有 FDboost 或只有一边点到的时段或配对，标成待复核，不写成确定影响。

## 参数改动时同步改什么

- 只加大模型容量（FDboost 的 `mstop` 上界、EBM 的 `max_rounds`、glinternet 的更小 `lambdaMinRatio`）时，选择本身仍由前向验证或 `validateFDboost` 决定。
- 改了入选特征个数（`num_feat`、`interactions`、`numToFind`）时，第 5 步的置换检验要重跑，旧的 p 值不能留。
- `GroupKFold` 的折数、前向块的切点、假发现率 0.10，四个方法共用，不要为一个方法单独放宽。

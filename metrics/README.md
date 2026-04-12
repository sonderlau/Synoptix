# Metrics

本目录存放项目中的评测指标实现，基于 `torchmetrics.Metric` 组织，并按“计算机制”拆分到了多个子目录中。
当前结构只保留分层后的子包实现，`metrics/` 根目录不再放置单个指标源码文件。

## 目录结构

```text
metrics/
├── core/           # 公共张量处理与内部基础工具
├── categorical/    # 阈值事件类指标，依赖 TP / FP / FN / TN
├── regression/     # 连续值误差类指标
├── perceptual/     # 图像感知质量类指标
├── neighborhood/   # 邻域 / 尺度容错类指标
├── object_based/   # 对象结构类指标
├── frequency/      # 频域类指标
├── distribution/   # 分布运输类指标
├── extremes/       # 峰值 / 极值类指标
├── __init__.py     # 顶层统一导出与别名
└── README.md
```

## 推荐导入方式

推荐使用两种方式：

```python
# 方式 1：从顶层导入常用别名，适合训练 / 评估脚本
from metrics import CSI, POD, FAR, SSIM, RMSE, FSS, SAL, EMD

# 方式 2：按类别导入，适合库代码和大项目
from metrics.categorical import CriticalSuccessIndex, ProbabilityOfDetection
from metrics.perceptual import StructuralSimilarityIndexMeasure
from metrics.neighborhood import FractionsSkillScore
```

注意：

- 不再支持 `from metrics.CriticalSuccessIndex import CriticalSuccessIndex` 这类旧路径。
- 现在统一使用 `from metrics import CSI` 或 `from metrics.categorical import CriticalSuccessIndex`。
- `metrics/__init__.py` 仍然提供常用顶层别名，例如 `CSI`、`FSS`、`SAL`、`EMD`、`SSIM`、`RMSE`。

## 分类与公式说明

### 1. `categorical/` 阈值事件类指标

这些指标先将连续场按阈值 `tau` 二值化，再基于混淆矩阵统计：

- `TP`：命中
- `FP`：误报
- `FN`：漏报
- `TN`：正确拒绝

实现说明：

- 该目录新增了 [`metrics/categorical/base.py`](/Users/sonderlau/Dev/Synoptix/metrics/categorical/base.py)，统一负责二值化、可选邻域池化、混淆矩阵累计和多阈值平均框架。
- 因此 `CSI`、`ETS`、`HSS`、`BIAS`、`POD`、`FAR` 的实现只保留各自的评分公式，避免重复状态管理代码。

#### `CriticalSuccessIndex`

- 类名：`CriticalSuccessIndex`
- 别名：`CSI`
- 公式：`CSI = TP / (TP + FP + FN)`
- 含义：衡量事件预测与观测事件的重叠程度，不考虑 `TN`

#### `CriticalSuccessIndexMean`

- 类名：`CriticalSuccessIndexMean`
- 别名：`CSIMean`
- 公式：`CSIMean = (1 / K) * sum_k CSI(tau_k)`
- 含义：对多个阈值下的 CSI 做平均

#### `HeidkeSkillScore`

- 类名：`HeidkeSkillScore`
- 别名：`HSS`
- 公式：
  `HSS = 2 * (TP * TN - FP * FN) / [ (TP + FN)(FN + TN) + (TP + FP)(FP + TN) ]`
- 含义：对“随机命中”做修正后的技能评分

#### `HeidkeSkillScoreMean`

- 类名：`HeidkeSkillScoreMean`
- 别名：`HSSMean`
- 公式：`HSSMean = (1 / K) * sum_k HSS(tau_k)`
- 含义：多个阈值下 HSS 的平均

#### `EquitableThreatScore`

- 类名：`EquitableThreatScore`
- 别名：`ETS`
- 公式：
  `ETS = (TP - TP_rand) / (TP + FP + FN - TP_rand)`

  `TP_rand = (TP + FN)(TP + FP) / N`

  `N = TP + FP + FN + TN`
- 含义：在 CSI 基础上扣除了随机命中的影响

#### `EquitableThreatScoreMean`

- 类名：`EquitableThreatScoreMean`
- 别名：`ETSMean`
- 公式：`ETSMean = (1 / K) * sum_k ETS(tau_k)`
- 含义：多个阈值下 ETS 的平均

#### `FrequencyBias`

- 类名：`FrequencyBias`
- 别名：`BIAS`
- 公式：`BIAS = (TP + FP) / (TP + FN)`
- 含义：预测事件频率与真实事件频率之比

#### `FrequencyBiasMean`

- 类名：`FrequencyBiasMean`
- 别名：`BIASMean`
- 公式：`BIASMean = (1 / K) * sum_k BIAS(tau_k)`
- 含义：多个阈值下 BIAS 的平均

#### `ProbabilityOfDetection`

- 类名：`ProbabilityOfDetection`
- 别名：`POD`
- 公式：`POD = TP / (TP + FN)`
- 含义：命中率 / 召回率，表示真实事件被检出的比例

#### `ProbabilityOfDetectionMean`

- 类名：`ProbabilityOfDetectionMean`
- 别名：`PODMean`
- 公式：`PODMean = (1 / K) * sum_k POD(tau_k)`
- 含义：多个阈值下 POD 的平均

#### `FalseAlarmRatio`

- 类名：`FalseAlarmRatio`
- 别名：`FAR`
- 公式：`FAR = FP / (TP + FP)`
- 含义：预测为事件的样本中，有多少实际上没有发生

#### `FalseAlarmRatioMean`

- 类名：`FalseAlarmRatioMean`
- 别名：`FARMean`
- 公式：`FARMean = (1 / K) * sum_k FAR(tau_k)`
- 含义：多个阈值下 FAR 的平均

### 2. `regression/` 连续值误差类指标

实现说明：

- 该目录保留最小而直接的实现，每个指标负责自己的误差定义与累计逻辑。

#### `MeanAbsoluteError`

- 类名：`MeanAbsoluteError`
- 别名：`MAE`
- 公式：`MAE = (1 / N) * sum_i |y_hat_i - y_i|`
- 含义：逐元素绝对误差平均值

#### `RootMeanSquaredError`

- 类名：`RootMeanSquaredError`
- 别名：`RMSE`
- 公式：`RMSE = sqrt((1 / N) * sum_i (y_hat_i - y_i)^2)`
- 含义：对大误差更敏感的均方根误差

### 3. `perceptual/` 图像感知质量类指标

#### `StructuralSimilarityIndexMeasure`

- 类名：`StructuralSimilarityIndexMeasure`
- 别名：`SSIM`
- 公式：
  `SSIM(x, y) = ((2 * mu_x * mu_y + C1)(2 * sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))`
- 含义：从亮度、对比度、结构三个层面衡量图像相似性

#### `PSNR`

- 名称：`PSNR`
- 公式：`PSNR = 10 * log10(L^2 / MSE)`
- 含义：峰值信噪比，`L` 为像素动态范围

### 4. `neighborhood/` 邻域 / 尺度容错类指标

实现说明：

- 邻域类指标共享 `core/` 中的张量展平逻辑，统一处理 `(B, T, H, W)` 到 `(B*T, H, W)` 的映射。

#### `FractionsSkillScore`

- 类名：`FractionsSkillScore`
- 别名：`FSS`
- 公式：
  先在窗口 `w` 内计算事件比例场 `f_w` 和 `o_w`

  `FSS_w = 1 - sum (f_w - o_w)^2 / (sum f_w^2 + sum o_w^2)`
- 含义：允许空间上有一定位移时，衡量预测场与观测场的相似程度

#### `FractionsSkillScoreMean`

- 类名：`FractionsSkillScoreMean`
- 别名：`FSSMean`
- 公式：`FSSMean = (1 / M) * sum_m FSS_{w_m}`
- 含义：对多个窗口尺度上的 FSS 做平均

### 5. `object_based/` 对象结构类指标

实现说明：

- 对象类指标与分布类指标共用 `core/` 的时空展平工具，减少维度预处理的重复代码。

#### `StructureAmplitudeLocation`

- 类名：`StructureAmplitudeLocation`
- 别名：`SAL`
- 公式：
  `SAL = {S, A, L}`

  `A = (sum pred - sum target) / sum target`

  `L = ||COM(pred) - COM(target)|| / d_max`

  `S`：本实现采用连通域面积分布熵差的近似形式来表征结构差异
- 含义：将误差拆成结构、振幅、位置三个部分，而不是压成一个单标量

#### `StructureAmplitudeLocationMean`

- 类名：`StructureAmplitudeLocationMean`
- 别名：`SALMean`
- 公式：分别对 `S`、`A`、`L` 分量在 batch 维度做平均
- 含义：输出仍然是一个包含 `structure`、`amplitude`、`location` 的字典

### 6. `frequency/` 频域类指标

#### `PowerSpectraDensityError`

- 类名：`PowerSpectraDensityError`
- 别名：`PSD`
- 公式：
  `P(k) = |FFT(x)|^2 / (H * W)`

  `PSD-Error = MSE(P_pred, P_target)`
- 含义：衡量预测和真实在频谱能量分布上的差异

### 7. `distribution/` 分布运输类指标

#### `EarthMoverDistance`

- 类名：`EarthMoverDistance`
- 别名：`EMD`
- 理论公式：
  `EMD(P, Q) = inf_{gamma in Pi(P, Q)} int ||x - y|| d gamma(x, y)`
- 当前实现说明：
  本实现不是严格最优传输求解器，而是用“质心距离 + 分布扩散差”的近似，并可按图像对角线归一化
- 含义：衡量将预测分布“搬运”为真实分布所需的代价

#### `Wasserstein1Distance`

- 类名：`Wasserstein1Distance`
- 别名：`Wasserstein`
- 公式：与 `EMD` 等价，即一阶 Wasserstein 距离
- 含义：当前实现直接复用 `EarthMoverDistance`

### 8. `extremes/` 峰值 / 极值类指标

#### `PeakValueRatio`

- 类名：`PeakValueRatio`
- 别名：`PVR`
- 公式：`PVR = max(pred) / max(target)`
- 含义：比较预测峰值与真实峰值的相对比例

## 使用示例

### 单个指标

```python
from metrics import SSIM

metric = SSIM()
score = metric(pred, target)
```

### 多指标组合

```python
from torchmetrics import MetricCollection
from metrics import CSI, FAR, POD, RMSE, SSIM

metrics = MetricCollection(
    {
        "rmse": RMSE(),
        "ssim": SSIM(),
        "csi_181": CSI(threshold=181),
        "pod_181": POD(threshold=181),
        "far_181": FAR(threshold=181),
    }
)

results = metrics(pred, target)
```

### 分类导入

```python
from metrics.categorical import CriticalSuccessIndex, FalseAlarmRatio
from metrics.regression import RootMeanSquaredError
from metrics.object_based import StructureAmplitudeLocation

csi = CriticalSuccessIndex(threshold=181)
far = FalseAlarmRatio(threshold=181)
rmse = RootMeanSquaredError()
sal = StructureAmplitudeLocation(threshold=0.0)
```

## 重构说明

这次重构只发生在 `metrics/` 目录内，目标是：

- 按计算类型组织文件，降低单层目录的认知负担
- 引入 `core/` 和分类内部基础模块，抽取重复的张量预处理与阈值统计逻辑
- 统一顶层导出，补齐别名，降低脚本侧导入成本
- 删除根目录旧指标源码文件，确保目录结构单一明确
- 在 README 中明确每个指标的公式、含义与推荐导入方式，减少二次查阅成本

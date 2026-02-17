# Metrics

本目录下为评测指标的实现，均基于 Torchmetrics。

## 已实现的指标

### 传统阈值类指标

- **CriticalSuccessIndex (CSI)**: 关键成功指数，衡量预测事件与观测事件的重叠程度
- **HeidkeSkillScore (HSS)**: 海德技能评分，考虑随机预测的技能评分
- **EquitableThreatScore (ETS)**: 公平威胁评分，考虑随机命中情况的评分
- **FrequencyBias (BIAS)**: 频率偏差，衡量预报事件频率与观测事件频率的比率
- **ProbabilityOfDetection (POD)**: 探测概率，即命中率或召回率
- **FalseAlarmRatio (FAR)**: 误报率，衡量预报事件中非实际发生的比例

### 图像质量指标

- **StructuralSimilarityIndexMeasure (SSIM)**: 结构相似性指数，衡量两幅图像的结构相似性
- **MeanAbsoluteError (MAE)**: 平均绝对误差
- **RootMeanSquaredError (RMSE)**: 均方根误差

### 邻域/尺度容错类指标

- **FractionsSkillScore (FSS)**: 分数技能评分，通过不同窗口大小计算降水分数场，反映尺度依赖的可预报性

### 对象/形状类指标

- **StructureAmplitudeLocation (SAL)**: 结构-振幅-位置评分，分解误差为结构、振幅和位置三个分量

### 频谱类指标

- **PowerSpectralDensityError (PSD)**: 功率谱密度误差，评估频域特征差异

### 分布类指标

- **EarthMoverDistance (EMD)**: 地球移动距离，衡量两个分布之间的最小传输成本
- **Wasserstein1Distance**: Wasserstein-1距离，与EMD等价

### 峰值类指标

- **PeakValueRatio (PVR)**: 峰值比率，评估预测与观测峰值的比率

## 使用方法

每个指标都继承自 Torchmetrics 的 Metric 类，可以像使用其他 PyTorch Lightning 指标一样使用：

```python
from metrics import SSIM, CSI, FSS, SAL, EMD

# 单个指标使用
ssim = SSIM()
result = ssim(predictions, targets)

# 批量使用多个指标
from torchmetrics import MetricCollection

metrics = MetricCollection([
    SSIM(),
    CSI(threshold=181),
    FSS(threshold=74, window_sizes=(3, 5, 7)),
    SAL(threshold=0.0),
    EMD()
])

results = metrics(predictions, targets)
```

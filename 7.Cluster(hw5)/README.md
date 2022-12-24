## 1. Requriments

* python 3.7
* numpy 1.21.5
* matplotlib 3.5.2
* scikit-learn  1.0.2
## 2. Dataset

[gmm](http://www.nustm.cn/member/rxia/ml/data/gmm.zip)
	   为了方便通过numpy导入数据，代码适配于将下载的txt文件的第一行删去

## 3. Start training

基于windows操作系统

GMM：
```powershell
python GMM.py --data_path GMM3
```

GMM聚类：

```
python GMMjulei.py --data_path GMM8
```

K-means

```
python K-means.py --data_path GMM8
```

如果想使用其他数据集，只需更换GMM8为GMM3或其他

## 4. Result

k-means在GMM8聚类结果

<img src='pic./Figure_2.png'>

GMM在GMM3上的结果

<img src='pic./Figure_1.png'>

更多结果可以通过运行运行程序获得

##### K-MEANS评价指标表

| K-means | Rand index | Silhouette Coeffcient | F-M scores |
| ------- | ---------- | --------------------- | ---------- |
| 3类     | 0.90       | 0.59                  | 0.93       |
| 4类     | 0.95       | 0.62                  | 0.96       |
| 6类     | 0.89       | 0.59                  | 0.91       |
| 8类     | 0.59       | 0.53                  | 0.67       |

##### 高斯混合聚类评价指标表

| GMM  | Rand index | Silhouette Coeffcient | F-M scores |
| ---- | ---------- | --------------------- | ---------- |
| 3类  | 0.91       | 0.60                  | 0.9        |
| 4类  | 0.95       | 0.62                  | 0.96       |
| 6类  | 0.74       | 0.54                  | 0.80       |
| 8类  | 0.54       | 0.39                  | 0.65       |

## 5 Analysis

通过上诉表格可以发现，两类聚类显然都是在聚类数量较少时效果更好，两种聚类方式效果差不多，仅在6类时K-means稍好于GMM。分析原因是可能k-means引入了非常强的人为假设，那就是距离越相近(本代码为欧式距离)那么两个样本就越可能属于同一类，而数据集并不完全遵循，在GMM8上尤其明显

<img src='pic./Figure_3.png' style="zoom:67%;" >

可以很清楚的发现青色与紫色有相交部分，那么也就不满足了k-means的假设。相比较gmm聚类，gmm聚类是soft assignment，可以得到属于每个类的概率，而k-means的hard assignment，只判断属于哪个类，所以结果不如gmm。
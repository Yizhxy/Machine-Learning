# Bayes

### Intro

##### 数据处理

由于数据集较大，设计的词较多，在初始化构造BOW的时候由于需要查询是否已经加入BOW导致运行很慢。

为了减少后续的查询时间，构造了word2index映射，可以通过输入word线性查找该word在BOW中的下标

##### 预测处理

贝叶斯最后是通过最大的后验概率得到模型最终的预测结果，但是在实际过程中，由于BOW太大，预测过程极其容易出现概率消失，所以进行以下处理
$$
\hat y =\arg\max\limits_{\ y}\ p(y|x)=\arg\max\limits_{\ y}\ \frac{p(x,y)}{\sum_{y}p(x,y)} \\
=\arg\max\limits_{\ y}\ p(x,y)=\arg\max\limits_{\ y}p(y)p(x|y) \\
=\arg\max\limits_{\ y}\pi*\prod \theta_{i,j}
=\arg\max\limits_{\ y}\pi*\prod K*\theta_{i,j}
$$
K是一个超参数，是一个常数

##### Requirements

* python 3.7
* numpy 1.21.5
* matplotlib 3.5.2
* libsvm

### Files

```java
--Tsinghua
    --train
    --test
--utils
    --init
    --eval
    --load
    --predict
    --word2index
--Bernoulli
--FW
--Multinomial
--svm
```

##### Start training

基于windows操作系统

多变量伯努利：

```powershell
python Bernoulli.py
```

多项式分布：

```powershell
python Multinomial.py
```

SVM(BOOL和TF)：

```powershell
python svm.py
```



### Result

|    method    |  ACC  |
| :----------: | :---: |
| 多变量伯努利 | 81.4% |
|  多项式分布  | 87.4% |
|  SVM(BOOL)   |  92%  |
|   SVM(TF)    |  72%  |


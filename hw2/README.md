# Logistic+Softmax

## Logistic

### 模型假设

$$
\delta(z)=\frac{1}{1+e^{-z}} \\
p(y=1|x;\theta)=h(x)=\delta(\theta^{T}x)\\
p(y=0|x;\theta)=1-h(x)
$$

统一形式
$$
p(y|x;\theta)=h(x)^{y}*(1-h(x))^{1-y}=(\frac{1}{1+e^{-\theta^T X}})^{y}(1-\frac{1}{1+e^{--\theta^T X}})^{1-y}
$$


### 最优化目标

由于模型的假设是通过条件概率的形式表达，那么很自然最优化的目标就是最大化似然函数
$$
L(\theta)=\prod_{k=1}^{N}p(y^{(k)}|x^{(k)};\theta)\\
=\prod_{k=1}^{N}h(x^{(k)})^{y^{(k)}}*(1-h(x^{(k)}))^{1-y^(k)}\\
=\prod_{k=1}^{N}=(\frac{1}{1+e^{-\theta^T x^{(k)}}})^{y^{(k)}}(1-\frac{1}{1+e^{-\theta^T x^{(k)}}})^{1-y^{(k)}}
$$
即
$$
\theta^{*}=\arg\max\limits_{\theta}L(\theta)=\arg\max\limits_{\theta}\log(L(\theta))\\
\log(L(\theta))=\sum_{k=1}^{N}y^{(k)}\log(h(x^{(k)}))*(1-y^{(k)})\log(1-h(x^{(k)}))
$$


### 学习算法

#### 梯度下降(上升)

#### 牛顿法

### 预测

## 实验

#### Requirements

* python 3.7
* numpy 1.21.5
* matplotlib 3.5.2

#### Dataset

[Exam dataset](http://www.nustm.cn/member/rxia/ml/data/Exam.zip)

[Iris](http://www.nustm.cn/member/rxia/ml/data/Iris.zip)

#### Files

```java
--data
    --Exam
      --test
    	  -x.txt
    	  -y.txt
      --train
          -x.txt
    	  -y.txt
    --Iris
      --test
    	  -x.txt
    	  -y.txt
      --train
          -x.txt
    	  -y.txt
--logistic_gd.py
--logistic_newton.py
--softmax.py
```

#### Start training

基于windows操作系统

##### Logistic:

GD：

```powershell
python logistic_gd.py --batch_size -1 --epoch 500
```

SGD:

```powershell
python logistic_gd.py --batch_size 16 --epoch 1000
```
Newton:
```powershell
python logistic_newton.py
```

##### Softmax

```powershell
python softmax.py
```

#### Result


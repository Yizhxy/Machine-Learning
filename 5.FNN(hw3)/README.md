# FNN

## 1 Intro

实现的层数有linear层，sigmoid层，softmax层

每一层都有前向传播和反向传播方法。

整个模型的前向传播过程就是依次执行forward函数，将上一层的输出作为下一层的输入，直到最后。

模型的反向传播过程则是依次执行bp函数，把后一层的梯度作为输入求该层的梯度。

因为作用的是分类任务，所以损失函数是交叉熵损失函数.

默认的模型结构是

<img src="model.png" alt="model" style="zoom:100%;" />

这是由代码中的init_model函数中的模块与参数确定。

<img src='init_model.png' style="zoom: 80%;" >

模型构成的层，层的类型，与层的参数都可以通过更改该函数修改.

注意

* 模型的输入一个与数据的特征数(*self.M*)相同
* 模型的输出与数据集的类别数(*self.L*)相同
* 隐藏层的输入大小应该与上一层的输出大小相同
## 2. Requriments

* python 3.7
* numpy 1.21.5
* matplotlib 3.5.2
* pytorch 1.10.1
* torchvision 0.11.2
## 3. Dataset

[Exam dataset](http://www.nustm.cn/member/rxia/ml/data/Exam.zip)

[Iris dataset](http://www.nustm.cn/member/rxia/ml/data/Iris.zip)

## 4. Start training

基于windows操作系统

Exam数据集

```powershell
python Fnn_myself.py --data_path ./data/Exam --batch_size 32 --epoch 1000
```

Iris数据集

```powershell
python Fnn_myself.py --data_path ./data/Iris --batch_size 32 --epoch 1000
```



## 5. Result

在Exam上的结果

<img src='Figure_2.png'>

在Iris上的结果

<img src='Figure_1.png'>

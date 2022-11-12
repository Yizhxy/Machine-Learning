##### Requirements

* python 3.7
* numpy 1.21.5
* matplotlib 3.5.2

##### Data

[Exam dataset](http://www.nustm.cn/member/rxia/ml/data/Exam.zip)

##### Files

```java
--data
    --Exam
      --test
    	  -x.txt
    	  -y.txt
      --train
          -x.txt
    	  -y.txt
--logistic_gd.py
```

##### Start training

基于windows操作系统

GD：

```powershell
python logistic_gd.py --batch_size -1 --epoch 500
```

SGD:

```powershell
python logistic_gd.py --batch_size 16 --epoch 1000
```


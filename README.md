# attention-seq2seq
***
### See for details [https://blog.csdn.net/qq_42714262/article/details/119298940](https://blog.csdn.net/qq_42714262/article/details/119298940)
***
### Requirements
python3.8<br>
pytorch==1.7.1 <br>
matplotlib==3.2.2
***
### how to use?
**1.train**<br/>
Just execute "train.py"<br>
**2.test**<br>
run the file "test.py"<br>
You can also modify the following code which is in the “test.py” to implement different inputs:<br>
```python
evaluateAndShowAttention("elle a cinq ans de moins que moi .")
evaluateAndShowAttention("elle est trop petit .")
evaluateAndShowAttention("je ne crains pas de mourir .")
evaluateAndShowAttention("c est un jeune directeur plein de talent .")
```
***
### Results
1. train loss <br>
![img.png](results/img.png)
2. attention matrix <br>
![img_1.png](results/img_1.png)![img_2.png](results/img_2.png)

# 说明
本项目目前为python或者c++实现的贝叶斯优化，代理函数使用高斯过程，虽然网上有部分贝叶斯优化的库，但是一般是针对于机器学习时超参数调参的优化，超参数的输入是一个确定的事件，但是现实生活中，可能一个函数、过程在实际运行时，会遇到环境的扰动，出于鲁棒性考虑，会取建议值附近的其他值，来保证完成任务，这时需要手动输入一对数据，本项目便增加了这一功能。

其中`svm_optimizer.py`为网上找的一份用于调超参数的代码，其余为本人写的代码。

## 更新
新增了只使用numpy版本高斯过程的的代码 `bayesian_opt_onlyNumpy.py` ，不需要`sklearn`库即可达到优化效果，但运行速度较慢。

## 更新
新增c++版本的贝叶斯优化，使用了eigen库。


### env install(windows)

#### 1. conda install
```
1. download conda from https://www.anaconda.com/
2. click the anaconda.exe to install
```

#### 2. create isolated environment
```
conda crate -n py-tf2x python=3.8
conda install cudatoolkit=10.1
conda install cudnn=7.6.5
```

#### 3. install dependency
```
conda activate py-tf2x
pip install tensorflow==2.4 -i https//pypi.douban.com/simple/
pip install matplotlib
pip instal sklearn
...
pip isntall [other dependency]
```

#### 4. test gpu is available
```
import tensorflow as tf
print(tf.test.is_gpu_available())
# print true, gpu env is ok
```


#### 5. attention
```
tensorflow-gpu version should compatible with cuda/cudnn version
you can find compatation at https://tensorflow.google.cn/install/source_windows#gpu
```
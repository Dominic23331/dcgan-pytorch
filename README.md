# DCGAN-pytorch

#### 所需环境
pytorch==1.9.0

torchvision==0.10.0


#### 生成步骤

1. 在DCGAN_core.py文件中修改DCGAN类中的defaults选项，将训练模型路径换为自己的模型路径

   ```python
   class DCGAN:
       _defaults = {
           "lr_d": 0.001,
           "lr_g": 0.001,
           "nz": 128,
           "nc": 3,
           "ngf": 64,
           "ndf": 64,
           "device": "cuda",
           "model_path": "你的模型文件路径"
       }
   ```

2. 运行generate.py进行生成图片

#### 训练步骤

1. 将数据集存放至./dataset/data路径中

2. 运行./dataset/data2csv.py文件，生成训练索引文件

3. 在train.py文件中设置训练参数，并进行训练

   ```python
   EPOCH = 100
   BATCH_SIZE = 128
   ```

#### 参考

1.  [PyTorch教程之DCGAN_我的学习笔记-CSDN博客](https://blog.csdn.net/weixin_36811328/article/details/88420820)
2.  [DCGAN Tutorial — PyTorch Tutorials 1.9.0+cu102 documentation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)


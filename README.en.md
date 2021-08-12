# DCGAN-pytorch

#### Requirements
pytorch==1.9.0

torchvision==0.10.0

#### Generate

1. Change your model path in defaults of class DCGAN

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

2. Run generate.py to generate image.

#### Train

1. Copy your dataset to ./dataset/data

2. Run /dataset/data2csv.py to generate training index file

3. Set the parameters in train.py and run this code.

   ```python
   EPOCH = 100
   BATCH_SIZE = 128
   ```

#### Reference

1.  [PyTorch教程之DCGAN_我的学习笔记-CSDN博客](https://blog.csdn.net/weixin_36811328/article/details/88420820)
2.  [DCGAN Tutorial — PyTorch Tutorials 1.9.0+cu102 documentation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)


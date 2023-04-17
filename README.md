# 项目说明
Pytorch实现联邦学习

# 环境
```
python 3.8.5
matplotlib==3.2.1
requests==2.28.2
torch==1.13.1
torchvision==0.9.1
PySocks==1.7.1
```

# 运行
* Multinode Multi-GPU:
在每台机器上执行
```
torchrun --nproc_per_node=6 --nnodes=10 --node_rank=0 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=x.x.x.x:xxxx main.py
torchrun --nproc_per_node=4 --nnodes=10 --node_rank=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=x.x.x.x:xxxx main.py
...... #在所有参与训练的机器上依次执行
```

* Single Node Multi-GPU:
在机器上执行
```
torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:xxxx main.py
```
关于`torchrun`中各个参数的意义， 可以参见[Pytorch文档](https://pytorch.org/docs/stable/elastic/run.html#definitions)

# Thanks
[Pytorch](https://pytorch.org/)
[Azure](https://azure.microsoft.com/)
[Zerotier](https://www.zerotier.com/)
[Code-server](https://coder.com/)
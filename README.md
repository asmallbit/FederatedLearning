# 项目说明
Pytorch实现联邦学习
https://github.com/asmallbit/FederatedLearning

# 环境
```
python 3.8.5
matplotlib==3.2.1
numpy==1.23.5
requests==2.28.2
torch==1.13.1
torchvision==0.14.1
PySocks==1.7.1
scikit-learn==1.2.2
```

# 目录结构
```
├── README.md
├── client.py
├── data				# 存放数据集资源文件
│   ├── cifar-10
│   └── mnist
├── dataset				# 自定义数据集
│   └── dog_and_cat.py
├── example				# 一些实例的配置文件
│   ├── cifar-resnet18-baseline.json
│   ├── cifar-resnet18.json
│   ├── mnist-mnist-cnn-baseline.json
│   └── mnist-mnist-cnn.json
├── figures				# 训练过程中可视化绘图
│   ├── cifar
│   └── mnist
├── main.py
├── main_mp.py			# CPU开启多个进程, 经测试仅适用于Linux, 对于Windows不适用
├── model				# 自定义的模型
│   ├── cat_and_dog.py
│   ├── mnist.py
│   └── simple_cnn.py
├── models.py
├── push				# 训练过程实时推送
│   ├── push.py
│   └── telegram
│       └── telegram_push.py
├── requirements.txt
├── result				# 训练后得到的模型
│   ├── cifar
│   └── mnist
├── server.py
└── utils				# 一些工具方法和配置文件
    ├── conf.json		# 程序配置文件
    ├── distributed_utils.py	# 一些关于Pytorch rpc的方法
    ├── output_handler.py		# 一些命令行输出和消息推送的方法
    └── utils.py				# 杂七杂八的方法
```

# 扩展此程序
扩展本程序，可以从数据集，模型以及推送方式等方面进行扩展。
如果要扩展数据集，可以在`./dataset/`文件夹下新建实现类，实现数据集的获取，在`./datasets.py`中添加数据集。
如果要扩展模型，可以在`./model/`文件夹下创建模型实现类，在`./models.py`中添加新模型。
如果要引入新的推送方式，可以在`./push`文件夹下实现信息的推送功能，在`./push/push.py`添加新的推送方式。

# 运行
Tips: 运行之前最好在各个参与训练的设备上先执行一下`torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:3214 main.py`，将模型和数据集下载到本地，因为涉及到的模式比较多，在不增加额外的配置变量的情况下，我目前还没有一个很好的方法来确定`Local Main Process`，导致本地各个进程都在同一时间对下载数据集和模型，产生多进程读写安全问题，进而可能导致程序运行失败。

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

* Single Node Multi-CPU:
在机器上执行
```
python3 main_mp.py --process_threshold $(nproc)		# process_threshold参数是要开启的进程数
```

关于`torchrun`中各个参数的意义， 可以参见[Pytorch文档](https://pytorch.org/docs/stable/elastic/run.html#definitions)

关于配置文件 `utils/conf.json`
```
{
	"model_name" : "resnet18",  // Model的类型, 可选值可以参见models.py文件
    
	"type" : "cifar",           // 数据集的类型，目前有cifar, mnist, dog-and-cat三个选项
	
	"global_epochs" : 50,       // 全局epoch

	"local_epochs" : 3,         // 本地epoch

	"alpha": 0.05,				// 刻画Non-IID程度

	"k": 3						// 聚类的簇数
	
	"batch_size" : 32,
	
	"lr" : 0.005,               // 学习率
	
	"lambda" : 0.1,

	"is_push_enable": true,     // 其否启用推送，接入推送的目的主要是方便实时查看训练的进度

	"push_type": "telegram",    // 推送方式，目前只接入了telegram

	"proxy_type": null,         // 是否需要使用代理，可选值socks5, http, https, 默认为null，即不使用代理

	"proxy_host": null,         // 主机名称，IP地址或者域名

	"proxy_port": null,         // 端口号

	"proxy_username": null,     // 代理验证的用户名

	"proxy_password": null,     // 代理验证的密码

	"telegram_id": "123456789", // 你的Telegram User id

	"api_key": "xxxxxxx:xxxxxxxxxx"     // Telegram Bot的token
}
```

# Special Thanks
[rtx4d](https://github.com/rtx4d) For his wonderful server:>

# Thanks
[OpenAI](https://openai.com/)
[0xc0de996](https://github.com/0xc0de996/Federated_Learning)
[Phani Rohith](https://towardsdatascience.com/federated-learning-through-distance-based-clustering-5b09c3700b3c)
[FederatedAI](https://github.com/FederatedAI/Practicing-Federated-Learning)
[Nutan](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118)
[Pytorch](https://pytorch.org/)
[Kaggle](https://www.kaggle.com)
[Google Colab](https://colab.research.google.com/)
[Oracle Cloud](https://www.oracle.com/cloud/)
[Azure](https://azure.microsoft.com/)
[Zerotier](https://www.zerotier.com/)
[Code-server](https://coder.com/)
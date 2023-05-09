# 项目说明
Pytorch实现联邦学习

# 环境
```
python 3.8.5
matplotlib==3.2.1
numpy==1.23.5
requests==2.28.2
torch==1.13.1
torchvision==0.9.1
PySocks==1.7.1
scikit-learn==1.2.2
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

关于配置文件 `utils/conf.json`
```
{
	"model_name" : "resnet18",  // Model的类型, 可选值可以参见models.py文件
    
	"type" : "cifar",           // 数据集的类型，目前有cifar, mnist, dog-and-cat三个选项
	
	"global_epochs" : 50,       // 全局epoch

	"local_epochs" : 3,         // 本地epoch

	"alpha": 0.05,				// 刻画Non-IID程度

	"k" : 3,					// 聚类的簇数
	
	"batch_size" : 32,
	
	"lr" : 0.005,               // 学习率

	"factor": 0.1,              // 客户端采用的自适应修改学习率， factor和patience是torch.optim.lr_scheduler.ReduceLROnPlateau()的两个参数

	"patience": 10,
	
	"momentum" : 0.9,        // SGD中的动量
	
	"lambda" : 0.1,

	"accuracy_difference_threshold" : 0.0001,   // 当两次全局的accuracy相差的绝对值小于此值，同时也满足loss_difference_threshold中的条件，会视为收敛，进而结束训练过程

	"loss_difference_threshold" : 0.0001,       // 当两次全局的loss相差的绝对值小于此值，和accuracy_difference_threshold搭配使用

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

# Thanks
[0xc0de996](https://github.com/0xc0de996/Federated_Learning)
[Pytorch](https://pytorch.org/)
[Azure](https://azure.microsoft.com/)
[Zerotier](https://www.zerotier.com/)
[Code-server](https://coder.com/)
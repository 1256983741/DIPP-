一、mczhi.github.io/DIPP/ 实验命令：

	conda activate torch113

	cd DIPP

	python data_process.py --load_path /path/to/original/data --save_path /output/path/to/processed/data --use_multiprocessing

	nohup python train.py --name DIPP --train_set ../train_set --valid_set ../valid_set --use_planning --pretrain_epochs 5 --train_epochs 20 --batch_size 32 --learning_rate 2e-4 --device cuda:0 > train.log 2>&1 &

	python open_loop_test.py --name open_loop --test_set /path/to/original/test/data --model_path /path/to/saved/model --use_planning --render --save --device cpu

	python closed_loop_test.py --name closed_loop --test_file /path/to/original/test/data --model_path /path/to/saved/model --use_planning --render --save --device cpu

二、实验前准备工作：

1.Google Cloud Storage下载数据集：

	gsutil -m cp -r \ " "\(下载文件夹地址)

2. install cuda

官网下载所需版本的 cuda[https://developer.nvidia.com/cuda-toolkit-archive](url)
版本根据实验需求选择

	wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run 

	sudo sh cuda_11.3.1_465.19.01_linux.run

终端出现窗口：
Do you accept the above EULA? (accept / decline / quit):

	accept

回车键进行勾选，X就是选中，没有X就是没有选中，把（driver）安装驱动进行取消。之后向下键，回车确认

最后点击 install

3. 配置cuda环境

	sudo  vim ~/.bashrc 

在bashrc文件最下方，添加下入代码
（ps：这边需要注意cuda的版本，版本不同，路径的命名需修改）

	export PATH=$PATH:/usr/local/cuda-11.8/bin

	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64

更新环境
	source ~/.bashrc

4.测试

	nvcc -V

输出下述结果，表示安装成功:
	nvcc: NVIDIA (R) Cuda compiler driver
	Copyright (c) 2005-2022 NVIDIA Corporation
	Built on Wed_Sep_21_10:33:58_PDT_2022
	Cuda compilation tools, release 11.3, V11.3.89
	Build cuda_11.3.r11.3/compiler.31833905_0

5.安装cudnn
下载cuda对应版本的cudnn包 https://developer.nvidia.com/rdp/cudnn-archive

将压缩包，放入自定义路径后，输入命令进行解压

	tar -xvf cudnn-linux-x86_64-8.2.0.53_cuda11-archive.tar.xz 

解压后，输入命令，讲cuDNN对应文件拷贝至CUDA指定路径

	cd cudnn-linux-x86_64-8.2.0.53_cuda11-archive/
	ls

	#include  lib  LICENSE

	sudo cp include/cudnn*.h /usr/local/cuda-11.3/include

	sudo cp lib/libcudnn* /usr/local/cuda-11.3/lib64

	sudo chmod a+r /usr/local/cuda-11.3/include/cudnn*.h /usr/local/cuda-11.3/lib64/libcudnn*

6.cuda版本切换

修改bashrc

	sudo vim ~/.bashrc

将原先的cuda-11.3注释掉，添加cuda-11.x新的环境设置，即可

	cuda-11.3
	export PATH=$PATH:/usr/local/cuda-11.3/bin
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64

	cuda-11.x
	export PATH=$PATH:/usr/local/cuda-11.1/bin
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.x/lib64

三、报错处理：

Theseus 安装问题（python 3.8，pytorch 1.12.1）ImportError：无法从 'torch.utils._pytree' 导入名称 'tree_map_only'
解决:更换pytorch版本（以下为测试版本）

torch113（已测试）
	conda create -n torch113 python=3.8 -y && conda activate torch113
	conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
	pip install theseus-ai==0.2.1

	？？？：UnpicklingError: Failed to interpret file '../train_set/8c480e8d57a313b9_110.npz' as a pickle（数据文件出问题？）
	
	AN：移除该数据
	
	？？？：Attempted to update variable control_variables with a (cuda:0,torch.float32) tensor, which is inconsistent with objective's expected (cuda,torch.float32).
	AN：训练数据的device选cuda:0（python train.py --name DIPP --train_set /path/to/train/data --valid_set /path/to/valid/data --use_planning --pretrain_epochs 5 --train_epochs 20 --batch_size 32 --learning_rate 2e-4 --device cuda：0）
	
	？？？：AttributeError: 'Objective' object has no attribute 'error_squared_norm'
	AN：修改train.py中“error_squared_norm()”函数为“error_metric()”函数 （新版theseus/core/objective.py删除了error_squared_norm()函数）


torch200（未测试）
	conda create -n torch200 python=3.8 -y && conda activate torch200
	conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
	pip install theseus-ai==0.2.1



begin training（指定epoch轮数）：
  add begin with 5 #第5轮报错，所以从第5轮重新训练
    > start_epoch = 5
	for epoch in range(start_epoch,train_epochs):
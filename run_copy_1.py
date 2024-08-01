import time
from torchviz import make_dot
import random
import numpy as np
# import matplotlib.pyplot as plt

import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data.data_yanlong import train_dataset, test_dataset
from models import MLP
from lib.trans_all import shaping
from lib import IK, IK_loss
import torch
import torch.nn as nn
import math
import os
from lib.save import checkpoints
from lib.plot import plot_IK_solution


class main():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training MLP")

        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 1)')
        self.parser.add_argument('--learning_rate', type=float, default=0.0025, help='learning rate (default: 0.003)')
        self.parser.add_argument('--epochs', type=int, default=10, help='gradient clip value (default: 300)')
        self.parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
        self.parser.add_argument('--num_train', type=int, default=500)

        self.args = self.parser.parse_args()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = TensorDataset(train_dataset.a[:self.args.num_train])
        self.data_loader_train = DataLoader(self.data, batch_size=self.args.batch_size, shuffle=False)

        # 定义训练权重保存文件路径
        self.checkpoint_dir = r'/home/cn/RPSN_2/work_dir/test01'

        self.num_i = 6
        self.num_h = 50
        self.num_o = 3
        
        # 如果是接着训练则输入前面的权重路径
        self.model_path = r''

        # 定义DH参数
        self.link_length = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
        self.link_offset = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
        self.link_twist = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])

        # # 记录所有epoch中四种类型的数量，用于后续画图
        # self.numError1 = []
        # self.numError2 = []
        # self.numNOError1 = []
        # self.numNOError2 = []
        # self.num_correct_test = []
        # self.num_incorrect_test = []
        # self.numPositionloss_pass = []
        # self.numeulerloss_pass = []



    def train(self):
        num_i = self.num_i
        num_h = self.num_h
        num_o = self.num_o

        epochs = self.args.epochs
        data_loader_train = self.data_loader_train

        learning_rate = self.args.learning_rate

        model = MLP.MLP_self(num_i , num_h, num_o) 
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.000)  # 定义优化器
        
        model_path = self.model_path

        if os.path.exists(model_path):          
            checkpoint = torch.load(model_path)  
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            print(f"The loading model is complete, let's start this training from the {start_epoch} epoch, the current loss is : {loss}")
        else:
            print("There is no pre-trained model under the path, and the following training starts from [[epoch1]] after random initialization")
            start_epoch = 1

        # 开始训练
        for epoch in range(start_epoch , start_epoch + epochs):
  
            sum_loss = 0.0

            for data in data_loader_train:  # 读入数据开始训练
                # a = 0
                inputs = data[0] # 取出每个batch_size的数据（默认五个）
                intermediate_outputs = model(inputs)
                input_tar = shaping(inputs) # 得到变换矩阵

                outputs = torch.empty((0, 6)) # 创建空张量
                for each_result in intermediate_outputs: # 取出每个batch_size中的每个数据经过网络后的结果1x3
                    pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
                    pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                    outputs = torch.cat([outputs, pinjie2.unsqueeze(0)], dim=0)

                intermediate_outputs.retain_grad()
                outputs.retain_grad()


                MLP_output_base = shaping(outputs)  # 对输出做shaping运算-1X6变为4X4
                MLP_output_base.retain_grad()

                # 计算 IK_loss_batch
                IK_loss_batch = torch.tensor(0.0, requires_grad=True)
                for i in range(len(inputs)):
                    angle_solution = IK.calculate_IK(
                        input_tar[i], 
                        MLP_output_base[i], 
                        self.link_length, 
                        self.link_offset, 
                        self.link_twist)
                    IK_loss_batch = IK_loss_batch + IK_loss.calculate_IK_loss(angle_solution)


                IK_loss_batch.retain_grad()

                optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0

                # 定义总loss函数
                loss = (IK_loss_batch) / len(inputs)
                loss.retain_grad()

                # 记录x轮以后网络模型checkpoint，用来查看数据流，路径选自己电脑的目标文件夹
                if epoch % 10 == 0:
                    print("第{}轮的网络模型被成功存下来了！储存内容包括网络状态、优化器状态、当前loss等".format(epoch))
                    checkpoints(model, epoch, optimizer, loss, self.checkpoint_dir)


                loss.backward()  # 反向传播求梯度
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip)  # 进行梯度裁剪
                optimizer.step()  # 更新所有梯度
                sum_loss = sum_loss + loss.data


            model.eval()

            data_test = TensorDataset(test_dataset.c, test_dataset.c)
            data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)
            for data_test in data_loader_test:
                with torch.no_grad():
                    inputs_test = data_test[0]
                    intermediate_outputs_test = model(inputs_test)
                    input_tar_test = shaping(inputs_test)
                    outputs_test = torch.empty((0, 6))  # 创建空张量
                    for each_result in intermediate_outputs_test:
                        pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
                        pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                        outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)

                    MLP_output_base_test = shaping(outputs_test)  # 对输出做shaping运算

                    # 计算 IK_loss_batch
                    IK_loss_batch_test = torch.tensor(0.0, requires_grad=True)
                    for i in range(len(inputs_test)):
                        angle_solution = IK.calculate_IK_test(
                            input_tar_test[i], 
                            MLP_output_base_test[i], 
                            self.link_length, 
                            self.link_offset, 
                            self.link_twist)
                        IK_loss_batch_test = IK_loss_batch_test + IK_loss.calculate_IK_loss_test(angle_solution, inputs_test[i], outputs_test[i])



            print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss / (len(data_loader_train))), "-" * 100)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            plot_IK_solution(1 , epochs)


if __name__ == "__main__":
    a = main()
    a.train()
    print(a)

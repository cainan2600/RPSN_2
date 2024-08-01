import time
from torchviz import make_dot
import random
import numpy as np
# import matplotlib.pyplot as plt

import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data.data_yanlong import train_dataset, test_dataset
from models import MLP
from lib.trans_all import *
from lib import IK, IK_loss, planner_loss
import torch
import torch.nn as nn
import math
import os
from lib.save import checkpoints
from lib.plot import *


class main():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training MLP")
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 1)')
        self.parser.add_argument('--learning_rate', type=float, default=0.0025, help='learning rate (default: 0.003)')
        self.parser.add_argument('--epochs', type=int, default=20, help='gradient clip value (default: 300)')
        self.parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
        self.parser.add_argument('--num_train', type=int, default=500)
        self.args = self.parser.parse_args()

        # 使用cuda!!!!!!!!!!!!!!!未补齐
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练集数据导入
        self.data_train = TensorDataset(train_dataset.a[:self.args.num_train])
        self.data_loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=False)
        # 测试集数据导入
        self.data_test = TensorDataset(test_dataset.c, test_dataset.c)
        self.data_loader_test = DataLoader(self.data_test, batch_size=self.args.batch_size, shuffle=False)

        # 定义训练权重保存文件路径
        self.checkpoint_dir = r'/home/cn/RPSN_2/work_dir/test'
        # 多少伦保存一次
        self.num_epoch_save = 10

        self.num_i = 12
        self.num_h = 100
        self.num_o = 6
        
        # 如果是接着训练则输入前面的权重路径
        self.model_path = r''

        # 定义DH参数
        self.link_length = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
        self.link_offset = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
        self.link_twist = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])

        # 定义放置位置
        self.ori_position = torch.FloatTensor([1.2258864965873641, -2.436513135828552, -0.8611439933798337, -2.6471777579888087, 4.037534027618943, -0.568570505349759]
)
    def train(self):
        num_i = self.num_i
        num_h = self.num_h
        num_o = self.num_o

        NUMError1 = []
        NUMError2 = []
        NUMNOError1 = []
        NUMNOError2 = []
        NUM_correct_test = []
        NUM_incorrect_test = []

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
            numError1 = 0
            numError2 = 0
            numNOError1 = 0
            numNOError2 = 0

            for data in data_loader_train:  # 读入数据开始训练
                # 将目标物体1x6与放置位置1x6组合为1x12
                inputs = shaping_inputs_6to12(self.ori_position, data[0])

                intermediate_outputs = model(inputs)

                # 将1x12输入转为10x1x6,
                input_tar = shaping_inputs_12to6(inputs) # 得到变换矩阵
                # 得到每个1x6的旋转矩阵
                input_tar = shaping(input_tar)
                # 将网络输出1x6转换为1x3
                intermediate_outputs = shaping_output_6to3(intermediate_outputs)

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

                for i in range(len(input_tar)):

                    IK_loss2 = torch.tensor(0.0, requires_grad=True)

                    angle_solution, num_Error1, num_Error2 = IK.calculate_IK(
                        input_tar[i], 
                        MLP_output_base[i], 
                        self.link_length, 
                        self.link_offset, 
                        self.link_twist)

                    # if not num_Error1 + num_Error2 == 0:
                    #     IK_y_o_n_tar = 0
                    # else:
                    #     IK_y_o_n_tar = 1

                    # 存在错误打印
                    numError1 = numError1 + num_Error1
                    numError2 = numError2 + num_Error2
                    # 计算单IK_loss
                    IK_loss1, num_NOError1, num_NOError2 = IK_loss.calculate_IK_loss(angle_solution)

                    # #计算plannerloss/目标物体位置和放置位置同时有解
                    # llll = int(len(input_tar) / 2)
                    # if i in range(llll):
                    #     angle_solution_ori, IK_y_or_n_ori = planner_loss.IK_yes_or_no(
                    #         input_tar[i + llll], 
                    #         MLP_output_base[i + llll], 
                    #         self.link_length, 
                    #         self.link_offset, 
                    #         self.link_twist)
                    #     if not IK_y_o_n_tar + IK_y_or_n_ori == 2:
                    #         IK_loss2 = IK_loss2 + torch.tensor([10])


                    # 总loss
                    IK_loss_batch = IK_loss_batch + IK_loss1 + IK_loss2

                    # 无错误打印
                    numNOError1 = numNOError1 + num_NOError1
                    numNOError2 = numNOError2 + num_NOError2
                                

                IK_loss_batch.retain_grad()

                optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0

                # 定义总loss函数
                loss = (IK_loss_batch) / len(inputs)
                loss.retain_grad()

                # 记录x轮以后网络模型checkpoint，用来查看数据流
                if epoch % self.num_epoch_save == 0:
                    # print("第{}轮的网络模型被成功存下来了！储存内容包括网络状态、优化器状态、当前loss等".format(epoch))
                    checkpoints(model, epoch, optimizer, loss, self.checkpoint_dir)


                # loss.backward()  # 反向传播求梯度
                loss.backward(torch.ones_like(loss))  # 反向传播求梯度
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip)  # 进行梯度裁剪
                optimizer.step()  # 更新所有梯度
                sum_loss = sum_loss + loss.data
            
            NUMError1.append(numError1)
            NUMError2.append(numError2)
            NUMNOError1.append(numNOError1)
            NUMNOError2.append(numNOError2)

            print("numError1", numError1)
            print("numError2", numError2)
            print("num_NOError1", numNOError1)
            print("num_NOError2", numNOError2)

            model.eval()

            data_loader_test = self.data_loader_test
            num_incorrect_test = 0
            num_correct_test = 0
            for data_test in data_loader_test:
                with torch.no_grad():
                    inputs_test = shaping_inputs_6to12(self.ori_position, data_test[0])
                    intermediate_outputs_test = model(inputs_test)
                    input_tar_test = shaping_inputs_12to6(inputs_test)
                    input_tar_test = shaping(input_tar_test)
                    intermediate_outputs_test = shaping_output_6to3(intermediate_outputs_test)
                    outputs_test = torch.empty((0, 6))
                    for each_result in intermediate_outputs_test:
                        pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
                        pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                        outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)

                    MLP_output_base_test = shaping(outputs_test)

                    # 计算 IK_loss_batch
                    IK_loss_batch_test = torch.tensor(0.0, requires_grad=True)
                    for i in range(len(inputs_test)):
                        angle_solution, IK_test_incorrect = IK.calculate_IK_test(
                            input_tar_test[i], 
                            MLP_output_base_test[i], 
                            self.link_length, 
                            self.link_offset, 
                            self.link_twist)
                        # IK时存在的错误打印
                        num_incorrect_test = num_incorrect_test + IK_test_incorrect

                        IK_loss_test, IK_loss_test_incorrect, IK_loss_test_correct = IK_loss.calculate_IK_loss_test(angle_solution, 
                                                                                                                    inputs_test[i], 
                                                                                                                    outputs_test[i])
                        # 计算IK_loss时存在的错误打印
                        num_incorrect_test = num_incorrect_test + IK_loss_test_incorrect
                        num_correct_test = num_correct_test + IK_loss_test_correct
                        # 计算IK_loss
                        IK_loss_batch_test = IK_loss_batch_test + IK_loss_test

            print("num_correct_test", num_correct_test)
            print("num_incorrect_test", num_incorrect_test)

            NUM_incorrect_test.append(num_incorrect_test)
            NUM_correct_test.append(num_correct_test)

            print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss / (len(data_loader_train))), "-" * 100)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # 画图
        plot_IK_solution(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUM_incorrect_test, NUM_correct_test)
        plot_train(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUMError1, NUMError2, NUMNOError1, NUMNOError2)

if __name__ == "__main__":
    a = main()
    a.train()
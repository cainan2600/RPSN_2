import torch
import math
from lib.find_closest import find_closest


inputs_of_final_result = []
outputs_of_MLP = []
final_result = []

# angle_solution传入ik运算的8组解或异常跳出的值，loss由此函数定义部分（总loss还有其他两部分）
def calculate_IK_loss(angle_solution):

    num_NOError1 = 0
    num_NOError2 = 0
    num_illegal = 0
    IK_loss = torch.tensor([0.0], requires_grad=True)
    legal_solution = []
    where_is_the_illegal = []
    if len(angle_solution) == 1:  # 判断是不是IK异常跳出的，如果是直接赋值给loss
        IK_loss = IK_loss + angle_solution

    else:
        # 不报错的IK运算有8组解，每组解6个关节值，这里的关节值可能是NaN
        for solution_index in range(8):
            ls = []
            for angle_index in range(6):
                if -math.pi <= angle_solution[solution_index][angle_index] <= math.pi:
                    ls.append(angle_solution[solution_index][angle_index])
                else:
                    num_illegal += 1
                    where_is_the_illegal.append([solution_index, angle_index])
                    
                    num_NOError2 += 1
                    
                    break
            if len(ls) == 6:
                legal_solution.append(ls)
                IK_loss = IK_loss + torch.tensor([0])
                break

        if num_illegal == 8:
            IK_loss = IK_loss + find_closest(angle_solution, where_is_the_illegal) #!!!!!优先惩罚nan产生项，loss定义在计算过程中

            num_NOError1 += 1

    return IK_loss, num_NOError1, num_NOError2

def calculate_IK_loss_test(angle_solution,aaaaaaaaaa, bbbbbbbbbb):

    IK_loss_test_incorrect = 0
    IK_loss_test_correct = 0

    aaaaaaaaaa = list(aaaaaaaaaa)

    num_illegal = 0
    IK_loss = torch.tensor([0.0], requires_grad=True)
    legal_solution = []
    where_is_the_illegal = []

    if len(angle_solution) == 1:  # 判断是不是IK异常跳出的，如果是直接赋值给loss
        IK_loss = IK_loss + angle_solution

    else:
        # 不报错的IK运算有8组解，每组解6个关节值，这里的关节值可能是NaN
        for solution_index in range(8):
            ls = []
            for angle_index in range(6):
                if -math.pi <= angle_solution[solution_index][angle_index] <= math.pi:
                    ls.append(float(angle_solution[solution_index][angle_index]))
                else:
                    num_illegal += 1
                    where_is_the_illegal.append([solution_index, angle_index])
                    break
            if len(ls) == 6:
                legal_solution.append(ls)

                IK_loss_test_correct += 1

                inputs_of_final_result.append(aaaaaaaaaa)
                outputs_of_MLP.append(bbbbbbbbbb)
                final_result.append(ls)
                IK_loss = IK_loss + torch.tensor([0])
                break

        if num_illegal == 8:
            IK_loss = IK_loss + find_closest(angle_solution, where_is_the_illegal)

            IK_loss_test_incorrect += 1

    # print(IK_loss)
    return IK_loss, IK_loss_test_incorrect, IK_loss_test_correct
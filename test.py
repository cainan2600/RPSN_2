import torch

list1 = torch.FloatTensor([1,2,3,4,5,6])
list2 = torch.FloatTensor([6,5,4,3,2,1])

new_list = torch.cat((list1, list2), -1)

print(new_list)

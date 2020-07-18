import torch

#在PyTorch中,同一种操作可能有很多种形式,下面用加法作为例子。
x=torch.rand(2,2) ; y=torch.zeros(2,2); z=torch.rand(2,2)
print(x+y,'\n')
print(torch.add(x,z),'\n')

#还可指定输出:
result=torch.empty(2,2)
torch.add(x,z,out=result)
print(result)
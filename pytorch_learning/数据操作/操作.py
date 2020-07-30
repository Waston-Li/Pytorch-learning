import torch

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
print(torch.cuda.get_device_name(0))
#－－－－－－－－－－－算术操作－－－－－－－－－－－
#在PyTorch中,同一种操作可能有很多种形式,下面用加法作为例子。
x=torch.rand(2,2) ; y=torch.zeros(2,2); z=torch.rand(2,2)
print(x+y,'\n')
print(torch.add(x,z),'\n')

#还可指定输出:
result=torch.empty(2,2)
torch.add(x,z,out=result)
print(result,'\n')

#adds x to y
y.add_(x)
print(y,'\n')

#PyTorch操作inplace版本都有后缀"", 例如`x.copy(y), x.t_()`

#－－－－－－－－－－－索引－－－－－－－－－－－
#我们还可以使用类似NumPy的索引操作来访问 Tensor  的一部分,
# 需要注意的是:索引出来的结果与原数据共享内存,也即修改一个,另一个会跟着修改

z=x[0:]
z+=1
print(z,'\n') # 源tensor也被改了
print(x[0:],'\n')
#除了常用的索引选择数据之外,PyTorch还提供了一些高级的选择函数:
#index_select(input, dim, index) 在指定维度dim上选取,比如选取某些行、某些列masked_select(input, mask) 例子如上,a[a>0],使用ByteTensor进行选取
#non_zero(input) 非0元素的下标gather(input, dim, index) 根据index,在dim维度上选取数据,输出的size与index一样

#－－－－－－－－－－－改变形状－－－－－－－－－－－
#  用 view() 来改变 Tensor 的形状:
y=x.view(4)
z=x.view(-1,2) # -1所指的维度可以根据其他维度的值推出来
print(x);print(y);print(z)
print(x.size(),y.size(),z.size(),'\n')
#注意 view()  返回的新tensor与源tensor共享内存(其实是同一个tensor),也即更改其中的一个,另外一个也会跟着改变。
# (顾名思义,view仅仅是改变了 对这个张量的观察⻆度 )

#所 以 如 果 我 们 想 返 回 一 个 真 正 新 的 副 本 ( 即 不 共 享 内 存 ) 该 怎 么 办 呢 ?
# Pytorch 还 提 供 了 一个 reshape()  可以改变形状,但是此函数并不能保证返回的是其拷⻉,所以不推荐使用。
# 推荐先用 clone 创造一个副本然后再使用 view
x_cp=x.clone().view(4)
x-=1
print(x)
print(x_cp,'\n')
#使用 clone 还有一个好处是会被记录在计算图中,即梯度回传到副本时也会传到源 Tensor

#另外一个常用的函数就是item() , 它可以将一个标量 Tensor 转换成一个Python number
z=torch.randn(1)    #只能为１
print(z,'--item--->',z.item(),'\n')

#－－－－－－－－－－－线性代数－－－－－－－－－－－
#另外,PyTorch还支持一些线性函数,这里提一下,免得用起来的时候自己造轮子,具体用法参考官方文档
#PyTorch中的 Tensor  支持超过一百种操作,包括转置、索引、切片、数学运算、线性代数、随机数等等,可参考官方文档

#－－－－－－－－－－－广播机制－－－－－－－－－－－
#面我们看到如何对两个形状相同的 Tensor  做按元素运算。
# 当对两个形状不同的 Tensor  按元素运算时,可能会触发广播(broadcasting)机制:
# 先适当复制元素使这两个 Tensor  形状相同后再按元素运算。
x = torch.arange(1, 3).view(1, 2)
print(x) 
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y,'\n')
#由于 x  和 y  分别是1行2列和3行1列的矩阵,如果要计算 x + y  ,
# 那么 x  中第一行的2个元素被广播(复制)到了第二行和第三行
# 而 y  中第一列的3个元素被广播(复制)到了第二列。如此,就可以对2 个3行2列的矩阵按元素相加。

#－－－－－－－－－－－运算的内存开销－－－－－－－－－－－
#y=x+y　会开辟新内存；可以用索引如　y[:]=x+y
# 我们还可以使用运算符全名函数中的 out  参数或者自加运算符 += (也即 add_() )达到上述效果,
# 例如torch.add(x, y, out=y) 和 y += x 及　y.add_(x) 。

#－－－－－－－－－－－TENSOR 和NUMPY相互转换－－－－－－－－－－－
#我们很容易用 numpy()  和 from_numpy()  将 Tensor  和NumPy中的数组相互转换。
# 但是需要注意的一点是: 这两个函数所产生的的 Tensor  和NumPy中的数组共享相同的内存(所以他们之间的转换很快),改变其中一个时另一个也会改变!!!

#使用 numpy() 将 Tensor 转换成NumPy数组:
a=torch.ones(3)
b=a.numpy()
print(a,'---＞　numpu: ',b)
a.add_(1)
print(a,'---＞　numpu: ',b,'\n')

#使用 from_numpy() 将NumPy数组转换成 Tensor
import  numpy as np
a=np.ones(2)
b=torch.from_numpy(a)
print('numpy: ',a,'--> ',b,'\n')

#还有一个常用的将NumPy中的array转换成 Tensor  的方法就是 torch.tensor() , 需要注意的是,
# 此方法总是会进行数据拷⻉(就会消耗更多的时间和空间),所以返回的 Tensor 和原来的数据不再共享内存


#TENSOR  ON GPU   用方法 to() 可以将 Tensor 在CPU和GPU(需要硬件支持)之间相互移动。
#print(b.device)
#b=b.to(device)
#print(b.device,'\n')

# c=torch.rand(2,2).cuda()
# print(c)
# print((b+c).device)

#动求梯度
x=torch.ones(2,2,requires_grad=True) #requires_grad
print(x,'\n',x.grad_fn,'\n')
y=x+2
print(y,'\n',y.grad_fn,'\n')
z=y*y*3
out=z.mean()
print(out,'\n')

#因为 out 是一个标量,所以调用 backward() 时不需要指定求导变量
out.backward()
print(x.grad,'\n')


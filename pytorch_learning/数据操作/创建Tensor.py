import torch
#创建一个5x3的未初始化的 Tensor
x=torch.empty(5,3)
print(x,'\n')

#创建一个5x3的随机初始化的 Tensor
x=torch.rand(5,3)
print(x,'\n')

#创建一个5x3的long型全0的 Tensor
x=torch.zeros(5,3,dtype=torch.long)
print(x,'\n')

#可以直接根据数据创建 注意[]
x=torch.tensor([5.5,3])
print(x,'\n')

#还可以通过现有的 Tensor  来创建,此方法会默认重用输入 Tensor  的一些属性,
# 例如数据类型,除非自定义数据类型。

# 返回的tensor默认具有相同的torch.dtype和torch.device
y=x.new_ones(5,3,dtype=torch.float64)
print(y)
# 指定新的数据类型
y=torch.randn_like(y,dtype=torch.float)
print(y,'\n')

#我们可以通过 shape 或者 size() 来获取 Tensor 的形状
print(x.size(),"  ",y.shape) #注意:返回的torch.Size其实就是一个tuple, 支持所有tuple的操作。

# 还有很多函数可以创建 Tensor ,去翻翻官方API就知道了,  dadasda
# 函数功能Tensor(*sizes) 基础构造函数tensor(data,) 类似np.array的构造函数ones(*sizes) 全1Tensor zeros(*sizes) 全0Tensor eye(*sizes) 对⻆线为1,其他为0 arange(s,e,step 从s到e,步⻓为step linspace(s,e,steps) 从s到e,均匀切分成steps份输出: 我们可以通过 shape 或者 size() 来获取 Tensor 的形状: 输出: 注意:返回的torch.Size其实就是一个tuple, 支持所有tuple的操作。
#这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu)





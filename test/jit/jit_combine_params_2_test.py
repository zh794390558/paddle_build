import paddle
print(paddle.__version__)
from paddle.jit import to_static
from paddle.static import InputSpec
paddle.set_device('cpu')

################## 模型组网 ##################
x_spec = [InputSpec([None, 4],dtype='float32')]
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.nn.Linear(4, 4)
        self.fc2 = paddle.nn.Linear(4, 4)
        self.bias = 0.4
        self.flag = paddle.ones([2],dtype="int32")
    ################## 导出函数 ① ##################
    @to_static(input_spec=x_spec)
    def forward(self, x):
        out = self.fc1(x)
        out = paddle.nn.functional.relu(out)
        out = paddle.mean(out)
        return out
    ################## 导出函数 ② ##################
    @to_static(input_spec=x_spec)
    def infer(self, input):
        out = self.fc2(input)
        out = out + self.bias
        out = paddle.mean(out)
        return out
    ################## 导出变量 ① ##################
    # For extra Python float
    @to_static(property=True)
    def fbias(self):
        return self.bias + 1
    ################## 导出变量 ② ##################
    # For extra Tensor
    @to_static(property=True)
    def fflag(self):
        return self.flag
    
################## 模型导出 ##################
net = Net()
paddle.jit.save(net, path="./export", use_combine=True)

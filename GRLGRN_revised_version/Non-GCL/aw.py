# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 定义自动加权损失
    awl = AutomaticWeightedLoss(2)


    # 定义一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)  # 用于分类
            self.fc3 = nn.Linear(20, 1)  # 用于回归

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            class_out = self.fc2(x)  # 分类任务输出
            reg_out = self.fc3(x)  # 回归任务输出
            return class_out, reg_out


    # 初始化模型、优化器和损失函数
    model = SimpleModel()
    optimizer = optim.Adam(list(model.parameters()) + list(awl.parameters()), lr=0.01)
    criterion_reg = nn.MSELoss()  # 回归损失
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失

    # 生成一些随机数据
    x_data = torch.randn(8, 10)  # 8个样本，每个样本10维
    y_reg = torch.randn(8, 1)  # 回归任务的真实值
    y_cls = torch.randint(0, 5, (8,))  # 8个分类任务标签，假设有5个类别

    # 训练步骤
    for epoch in range(100):
        optimizer.zero_grad()

        # 前向传播
        class_pred, reg_pred = model(x_data)

        # 计算单独的损失
        loss_reg = criterion_reg(reg_pred, y_reg)
        loss_cls = criterion_cls(class_pred, y_cls)

        # 使用自动加权损失
        loss = awl(loss_reg, loss_cls)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印损失
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Total Loss = {loss.item():.4f}, Loss_reg = {loss_reg.item():.4f}, Loss_cls = {loss_cls.item():.4f}")

    # 输出最终的权重参数
    print("Final loss weights:", awl.params.detach().numpy())




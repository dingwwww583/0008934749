import torch
from torch import nn, optim
from data_preprocess import data_loader_MNIST
from tqdm import tqdm
import os

# 设置设备 (自动检测GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据加载
train_loader, test_loader = data_loader_MNIST()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加Dropout防止过拟合
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


# 初始化模型、损失函数和优化器
model = MyModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)  # 学习率调度器

# 训练参数
num_epochs = 15
best_accuracy = 0.0

# 创建保存目录
os.makedirs('checkpoints', exist_ok=True)

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [训练]')
    for inputs, labels in train_progress:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # 更新进度条
        train_progress.set_postfix(loss=loss.item(), accuracy=train_correct / train_total)

    # 计算平均训练损失和准确率
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_correct / train_total

    # 测试阶段
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        test_progress = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [测试]')
        for inputs, labels in test_progress:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_progress.set_postfix(loss=loss.item(), accuracy=test_correct / test_total)

    # 计算测试集指标
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total

    # 更新学习率
    scheduler.step(test_accuracy)

    # 打印结果
    print(f"\nEpoch {epoch + 1}/{num_epochs} | "
          f"训练损失: {avg_train_loss:.4f} | 训练准确率: {train_accuracy:.4f} | "
          f"测试损失: {avg_test_loss:.4f} | 测试准确率: {test_accuracy:.4f} | "
          f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': test_accuracy
        }, f'checkpoints/best_model.pth')
        print(f"保存最佳模型，准确率: {test_accuracy:.4f}")

print(f"\n训练完成！最佳测试准确率: {best_accuracy:.4f}")
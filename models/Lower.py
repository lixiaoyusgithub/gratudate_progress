import torch.utils.data
from torch import nn, optim
from torchvision.transforms import transforms
from Image_Processing.GaitDataset import P_GaitDataset, N_GaitDataset
from torch.utils.data import DataLoader, random_split
from model import LowLevelFeatureExtractor
import matplotlib.pyplot as plt

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def train_model(model, train_dataloader, criterion, optimizer, num_epochs):
    train_accuracies = []  # 用于记录每个epoch的训练准确度
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_dataloader:
            # 将数据移动到 GPU 上
            images = images.to(device)
            labels = labels.to(device)

            #消除梯度
            optimizer.zero_grad()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total
        train_accuracies.append(epoch_acc)  # 记录当前epoch的训练准确度
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

    return train_accuracies


def test_model(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            # 将数据移动到 GPU 上
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()  # 去除数组中维度大小只有1的维度
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    print(f'Test Acc: {test_acc:.4f}')


if __name__ == "__main__":
    p_root_dir = r'D:\研究生课程\研究生课程\database\GAIT-IST'
    n_root_dir = r'D:\研究生课程\研究生课程\database\_BDataset'
    p_dataset = P_GaitDataset(p_root_dir, transform=transform)
    n_dataset = N_GaitDataset(n_root_dir, transform=transform)

    # 确保n_dataset的长度是p_dataset的三倍
    p_length = len(p_dataset)
    n_length = len(n_dataset)
    print(p_length, n_length)
    if n_length > 3 * p_length:
        n_dataset = torch.utils.data.Subset(n_dataset, range(3 * p_length))

    all_dataset = torch.utils.data.ConcatDataset([p_dataset, n_dataset])
    train_size = int(0.8 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = LowLevelFeatureExtractor()
    # 将模型移动到 GPU 上
    model = model.to(device)
    criterion = nn.BCELoss()
    # 将损失函数移动到 GPU 上
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # 训练轮数
    # 训练模型
    train_accuracies = train_model(model, train_dataloader, criterion, optimizer, num_epochs)
    # 在测试集上评估模型
    test_model(model, test_dataloader)

    # 绘制训练准确度折线图
    plt.plot(range(1, num_epochs + 1), train_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.grid(True)
    plt.show()
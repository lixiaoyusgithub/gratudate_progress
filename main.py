import torch.utils.data
from torch import nn, optim
from torchvision.transforms import transforms
from GAIT_DataSet import GAIT_DataSet
from torch.utils.data import DataLoader, random_split
from model import FeatureFusionModel
import matplotlib.pyplot as plt
import os
import time

# 检查 GPU 是否可用，若可用则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 创建 Logs 文件夹
logs_dir = 'Logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 生成唯一的文件名，使用时间戳
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(logs_dir, f'training_log_{timestamp}.txt')
image_file = os.path.join(logs_dir, f'training_curves_{timestamp}.png')

# 每个批次加载的数据样本数量
batch_size = 32
# 定义图像预处理的转换操作
transform = transforms.Compose([
    # 将图像大小调整为 224x224
    transforms.Resize((224, 224)),
    # 将图像转换为张量
    transforms.ToTensor(),
    # 对二值图像进行归一化处理
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 训练模型的函数
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, log_file):
    # 用于记录每个 epoch 的训练准确度、训练损失、测试准确度和测试损失
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    with open(log_file, 'a') as f:
        for epoch in range(num_epochs):
            # 将模型设置为训练模式
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_dataloader:
                # 将数据移动到 GPU 上
                images = images.to(device)
                labels = labels.to(device)

                # 清零优化器的梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = model(images)
                # 计算损失
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()
                # 更新模型参数
                optimizer.step()

                running_loss += loss.item()
                # 根据输出进行预测
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 计算当前 epoch 的训练损失和准确率
            epoch_train_loss = running_loss / len(train_dataloader)
            epoch_train_acc = correct / total
            train_accuracies.append(epoch_train_acc)
            train_losses.append(epoch_train_loss)

            # 在测试集上评估模型
            epoch_test_loss, epoch_test_acc = test_model(model, test_dataloader, criterion)
            test_accuracies.append(epoch_test_acc)
            test_losses.append(epoch_test_loss)

            log_info = f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}'
            print(log_info)
            f.write(log_info + '\n')

    return train_accuracies, train_losses, test_accuracies, test_losses

# 测试模型的函数
def test_model(model, test_dataloader, criterion):
    # 将模型设置为评估模式
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            # 将数据移动到 GPU 上
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            # 根据输出进行预测
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算测试集的损失和准确率
    test_loss = running_loss / len(test_dataloader)
    test_acc = correct / total
    return test_loss, test_acc

# 绘制曲线的函数
def plot_curves(train_accuracies, train_losses, test_accuracies, test_losses, num_epochs, image_file):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()

    plt.tight_layout()
    # 保存图片
    plt.savefig(image_file)
    plt.show()

if __name__ == "__main__":

    #GAIT-IST数据集实验
    normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IST_Extract_paths.json'
    parkinsonian_root_dir =r'D:\研究生课程\graduate_title\Jsons\P_IST_Extract_paths.json'

    #normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IST_GEIs_front.json'
    #parkinsonian_root_dir = r'D:\研究生课程\graduate_title\Jsons\P_IST_GEIs_front.json'

    #normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IST_silhouettes_front.json'
    #parkinsonian_root_dir = r'D:\研究生课程\graduate_title\Jsons\P_IST_silhouettes_front.json'

    #normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IST_silhouettes_back.json'
    #parkinsonian_root_dir = r'D:\研究生课程\graduate_title\Jsons\P_IST_silhouettes_back.json'


    #自提取能量图
    #normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IST_Extract_paths_byour_front.json'
    #parkinsonian_root_dir = r'D:\研究生课程\graduate_title\Jsons\P_IST_Extract_paths_byour_front.json'



    #GAIT-IT数据集实验
    # normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IT_GEIs_back.json'
    # parkinsonian_root_dir =r'D:\研究生课程\graduate_title\Jsons\P_IT_GEIs_back.json'

    # normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IT_GEIs_front.json'
    # parkinsonian_root_dir = r'D:\研究生课程\graduate_title\Jsons\P_IT_GEIs_front.json'

    # normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IT_silhouettes_front.json'
    # parkinsonian_root_dir = r'D:\研究生课程\graduate_title\Jsons\P_IT_silhouettes_front.json'

    # normal_root_dir = r'D:\研究生课程\graduate_title\Jsons\N_IT_silhouettes_back.json'
    # parkinsonian_root_dir = r'D:\研究生课程\graduate_title\Jsons\P_IT_silhouettes_back.json'


    #
    # 将 JSON 文件名称写入日志
    with open(log_file, 'a') as f:
        f.write(f"Normal JSON file: {os.path.basename(normal_root_dir)}\n")
        f.write(f"Parkinsonian JSON file: {os.path.basename(parkinsonian_root_dir)}\n")

    # 创建数据集对象
    dataset = GAIT_DataSet(normal_root_dir, parkinsonian_root_dir, transform=transform)

    # 计算训练集的大小
    train_size = int(0.8 * len(dataset))
    # 计算测试集的大小
    test_size = len(dataset) - train_size
    # 划分训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建训练集的数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    # 创建测试集的数据加载器
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # 创建特征融合模型
    fusion_model = FeatureFusionModel(use_transformer=False).to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    # 定义优化器
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
    # 定义训练轮数
    num_epochs = 50

    # 训练融合模型
    train_accuracies, train_losses, test_accuracies, test_losses = train_model(fusion_model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, log_file)

    # 绘制曲线并保存图片
    plot_curves(train_accuracies, train_losses, test_accuracies, test_losses, num_epochs, image_file)
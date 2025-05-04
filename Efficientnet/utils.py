import os
import sys
import random
import csv
import torch
from tqdm import tqdm

def split_data(root: str, val_rate: float, the_line: int):
    assert os.path.exists(root), "训练文件夹填错了"

    label_path = os.path.join(root, "label.csv")
    assert os.path.exists(label_path), "label.csv文件不存在"
    with open(label_path, mode='r', encoding='utf-8') as file:
        all_label = []
        csv_reader = csv.reader(file)
        for row in csv_reader:
            all_label.append(row)

    image_root = os.path.join(root, "image")
    assert os.path.exists(image_root), "image文件夹不存在"
    # 沟槽的17号发个花的图片上来污染数据集
    image_list = [i for i in os.listdir(image_root) if os.path.basename(i) != "17.jpg"]

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    
    val_list = random.sample(image_list, int(len(image_list) * val_rate))

    for img_path in image_list:
        if img_path in val_list:
            val_images_path.append(os.path.join(image_root, img_path))
            i, _ = os.path.splitext(img_path)
            image_name = os.path.basename(i)
            val_images_label.append(int(all_label[int(image_name)][2]))
        else:
            train_images_path.append(os.path.join(image_root, img_path))
            j, _ = os.path.splitext(img_path)
            image_name = os.path.basename(j)
            train_images_label.append(int(all_label[int(image_name)][2]))

    return train_images_path, train_images_label, val_images_path, val_images_label

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num


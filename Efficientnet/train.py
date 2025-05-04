import os
import sys
import json
import math
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from my_dataset import MyDataSet
from model import efficientnet_b7 as creat_model
from utils import split_data, train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter

def main():
    root = r"C:\somefiles\handdataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    tb_writer = SummaryWriter(log_dir=r"runs/efficientnet_b7")

    image_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    model_type = image_size["B7"]

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(model_type),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(model_type),
                                   transforms.CenterCrop(model_type),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 最后一个参数是预测目标，性别1， 爱情线2，事业线3，生命线4
    train_images_path, train_images_label, val_images_path, val_images_label = split_data(root, 0.2, 2)

    model_name = "Efficientnet_b7"
    model = creat_model(num_classes=2)# 预测类别数
    model.to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, model_type, model_type), device=device)
    tb_writer.add_graph(model, init_img)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    

    pre_weights = r"C:\somefiles\handdataset\efficientnetb7.pth"

    if pre_weights:
        # 迁移学习
        if os.path.exists(pre_weights):
            weights_dict = torch.load(pre_weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
        else:
            raise FileNotFoundError("not found weights file: {}".format(pre_weights))

    for name, para in model.named_parameters():
        # 除最后一个卷积层和全连接层外，其他权重全部冻结
        if ("features.top" not in name) and ("classifier" not in name):
            para.requires_grad_(False)
        else:
            print("training {}".format(name))
    
    lr = 0.01 # 学习率
    lrf = 0.01 # 最小学习率
    epochs = 50 # 训练轮数

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # 余弦退火（cosine annealing）方法
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) # 根据lambda函数调整学习率

    best_acc = 0.0
    min_mean_loss = 0.350


    for epoch in range(epochs):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        save_path_acc = "C:\\somefiles\\handdataset\\{}Net_{}_acc.pth".format(model_name, epoch)
        save_path_loss = "C:\\somefiles\\handdataset\\{}Net_{}_loss.pth".format(model_name, epoch)     
        if acc >= best_acc and epoch > 4: # 不存前五次的
            best_acc = acc + 0.01 # 验证集太少了，基本每次都是一样的，只存第一次的
            print("The best accuracy is {:.3f}, save weights to {}".format(best_acc, save_path_acc))
            torch.save(model.state_dict(), save_path_acc)

        if mean_loss <= min_mean_loss:
            min_mean_loss = mean_loss
            print("The min_mean_loss is {:.3f}, save weights to {}".format(min_mean_loss, save_path_loss))
            torch.save(model.state_dict(), save_path_loss)

if __name__ == "__main__":    
    main()

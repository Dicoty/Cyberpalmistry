import os
import torch
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from model import efficientnet_b3 as creat_model
import matplotlib.pyplot as plt
from my_dataset import MyDataSet
from utils import split_data

def main():
    root = r""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    model_type = image_size["B3"]

    data_transforms = transforms.Compose([transforms.Resize(model_type),
                                   transforms.CenterCrop(model_type),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    model = creat_model(num_classes=2)
    model.to(device)
    weights = r""
    weights_dict = torch.load(weights, map_location=device, weights_only=True)

    # 最后一个参数是预测目标，性别1， 爱情线2，事业线3，生命线4
    train_images_path, train_images_label, val_images_path, val_images_label = split_data(root, 1, 2)

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transforms)
    
    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    
    model.load_state_dict(weights_dict, strict = False)

    with torch.no_grad():
        model.eval()

        total_num = len(val_loader.dataset)

        TP = torch.zeros(1).to(device)
        FP = torch.zeros(1).to(device)
        FN = torch.zeros(1).to(device)
        TN = torch.zeros(1).to(device)

        data_loader = tqdm(val_loader, file = sys.stdout)

        for step, data in enumerate(data_loader):
            images, labels = data
            pred = model(images.to(device))
            pred = torch.max(pred, dim=1)[1]
            # print(pred)
            # print(labels)
            for i in range(labels.numel()):
                if pred[i].item() == 1 and labels[i].item() == 1:
                    TP += 1
                elif pred[i].item() == 1 and labels[i].item() == 0:
                    FP += 1
                elif pred[i].item() == 0 and labels[i].item() == 0:
                    TN += 1
                elif pred[i].item() == 0 and labels[i].item() == 1:
                    FN += 1
    
    plot_confusion_matrix(TP.item(), FP.item(), FN.item(), TN.item())


def plot_confusion_matrix(TP, FP, FN, TN):
    plt.figure(figsize=(8, 8))
    confusion_matrix = [[int(TP),int(FP)],[int(FN),int(TN)]]
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    thresh = np.max(confusion_matrix) / 2.0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            plt.text(j, i, format(confusion_matrix[i][j], 'd'),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i][j] > thresh else "black")
    plt.show()


if __name__ == '__main__':
    main()
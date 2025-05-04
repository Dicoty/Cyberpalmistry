import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import efficientnet_b3 as creat_model
import matplotlib.pyplot as plt

def main():
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
    
    img_path = r""
    assert os.path.exists(img_path), "Image is not exists"
    img = Image.open(img_path)

    plt.imshow(img)
    img = data_transforms(img)
    img = torch.unsqueeze(img, dim=0)
    
    model = creat_model(num_classes=2).to(device)
    class_indices = {'0': "no", '1': "yes"}
    model.load_state_dict(torch.load(r"C:\somefiles\handdataset\Efficientnet_b3Net_26_acc_0.8.pth", weights_only=True))

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(predict)
    print_res = "class: {}   prob: {:.3}".format(class_indices[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indices[str(i)],
                                                  predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()
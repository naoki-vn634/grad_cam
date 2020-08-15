import torch
import cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision.transforms import transforms

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def toHeatmap(x):
    x = (x*255).reshape(-1)
    cm = plt.get_cmap('jet')
    x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
    return x.reshape(224,224,3)

def GradCam(img, feature_fn, classifier_fn):
    middle = feature_fn(img)
    n, c, h, w = middle.size()
    output = classifier_fn(middle)
    score = output[0, torch.max(output, 1)[1]]
    grads = torch.autograd.grad(score, middle)
    weight = grads[0][0].mean(-1).mean(-1)  # 最初のindexはtupleの内，勾配を抽出する
    sal = F.relu(torch.matmul(weight, middle.view(c, h*w)))
    sal = sal.view(h,w).cpu().detach().numpy()
    sal = np.maximum(sal, 0)

    return sal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.vgg16(True).to(device)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

img_path = 'cat.jpg'
img = cv2.imread(img_path)
img_transform = transforms(img).unsqueeze(0).to(device)
print(img.shape)

feature_fn = torch.nn.Sequential(*list(model.children())[:-2]).to(device)
classifier_fn = torch.nn.Sequential(*list(model.children())[-2: -1] + [Flatten()] + list(model.children())[-1:]).to(device)

middle = feature_fn(img_transform)
out = classifier_fn(middle)
sal = GradCam(img_transform, feature_fn, classifier_fn)

# img_sal = np.array(Image.fromarray(sal).resize(img.shape[:2],resample=Image.LINEAR))
sal_norm = (sal/np.max(sal))
L = toHeatmap(cv2.resize(sal_norm, (224,224)))
L = (L/np.max(L))*255

alpha = 0.3
blended = img*alpha + L*(1-alpha)

cv2.imwrite('result.png', blended)
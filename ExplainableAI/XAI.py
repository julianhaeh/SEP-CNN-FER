import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from Data.clsOurDatasetSCN import OurDatasetSCN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import random
from Data.clsOurDataset import OurDataset
from sklearn.decomposition import PCA
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from ModelArchitectures.clsDownsizedCustomVGG13Reduced import DownsizedCustomVGG13Reduced
from ModelArchitectures.clsMobileFaceNet import MobileFacenet
from ModelArchitectures.clsGAP_4_64_96_128_196 import GAP_4_64_96_128_196
from ModelArchitectures.clsReducedClassifierCustomVGG13Reduced import ReducedClassifierCustomVGG13Reduced
from ExplainableAI.GradCAM import OurGradCAM
from ExplainableAI.SaliencyMaps import OurSaliencyMaps
from ExplainableAI.FeatureMaps import OurFeatureMaps
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
# 0 = Angry / 1 = Disgust / 2 = Fear / 3 = Happy / 4 = Sad / 5 = Surprise
dataset = OurDataset(split='test')
sample = [dataset[i+20] for i in range(20)]
images = [s["image"].unsqueeze(0) for s in sample]
labels = [s["label"] for s in sample]
img = images[1]
label = labels[1]
def get_random_images(label = None, amount = 10):
    count = random.randint(0, 1000)
    random_data = []
    i = 0
    if label is None:
        while (i<amount):
            random_data.append(dataset[i+count])
            i+=1
    else:
        while (i<amount):
            if (dataset[count + i]["label"] == label):
                random_data.append(dataset[i+count])
                i+=1
            else:
                count +=1
    random_images = [s["image"].unsqueeze(0) for s in random_data]
    random_labels = [s["label"] for s in random_data]
    random_images_and_labels = (random_images, random_labels)
    return random_images_and_labels
Original = CustomVGG13Reduced()
weights1 = torch.load("Experiments/Models/VGG13_Weighted_CE_Acc_72.30_Model.pth", map_location=torch.device('cpu'))
Original.load_state_dict(weights1)
# print(weights1.keys())
# print(model1)

Downsized = DownsizedCustomVGG13Reduced()
weights2 = torch.load("Experiments/Models/CustomVGG13_Downsized_Acc_72.51_Model.pth", map_location=torch.device('cpu'))
Downsized.load_state_dict(weights2)
Downsized.eval()
#print(model2)
# print(weights2.keys())

ReducedClassifier = ReducedClassifierCustomVGG13Reduced()
weights4 = torch.load("Experiments/Models/ReducedClassifier_Weighted_CE_Weighted_Acc_72.84_Model.pth", map_location=torch.device('cpu'))
ReducedClassifier.load_state_dict(weights4)
ReducedClassifier.eval()
print(ReducedClassifier)
print(weights4.keys)

GAP = GAP_4_64_96_128_196()
weights4 = torch.load("Experiments/Models/GAP_4_64_96_128_196_Weighted_CE_Weighted_Acc_72.10_Model.pth", map_location=torch.device('cpu'))
GAP.load_state_dict(weights4)
GAP.eval()

MobileFacenet = MobileFacenet()
weights5 = torch.load("mobilefacenet_20260201_123158.pth", map_location=torch.device('cpu'))
modelweights = weights5["model_state_dict"]
headweights = weights5["head_state_dict"]
MobileFacenet.load_state_dict(modelweights)
MobileFacenet.eval()
head = ClassificationHead(128, 6)
head.load_state_dict(headweights)
head.eval()

# print("Original: ", Original)
# print("Downsized: ", Downsized)
# print("ReducedClassifier: ", ReducedClassifier)
# print("MobileFacenet: ", MobileFacenet)
# print("GAP: ", GAP)
with torch.no_grad():
    emb = MobileFacenet(img)
    output = head(emb)
print("Prediction: ", F.softmax(output, dim = 1))
print(label)

test_model = ReducedClassifier

with torch.no_grad():
    output = test_model(img)
print("Prediction: ", F.softmax(output, dim = 1))
print(label)

TestGradCAM = OurGradCAM(test_model)
imgs, lbls = get_random_images(label = 5, amount=20)
TestGradCAM.GradCAMmany(imgs, 7)
TestGradCAM.GradCAMmany(imgs, 8)
TestGradCAM.GradCAMmany(imgs, 9)
"""
TestFeatureMaps = OurFeatureMaps(test_model)
TestFeatureMaps.FeatureMaps(imgs[0], 8)
TestFeatureMaps.get_topk_feature_maps(imgs[0], test_model.features[27], k = 16, metric = "max")
TestFeatureMaps.FeatureMaps(imgs[0], 9)
TestFeatureMaps.get_topk_feature_maps(imgs[0], test_model.features[30], k = 16, metric = "max")

TestGradCAM.GradCAM(img)
TestSaliencyMaps = OurSaliencyMaps(test_model)
TestSaliencyMaps.SaliencyMap(img)


with torch.no_grad():
    output = test_model(img)
print("Prediction: ", F.softmax(output, dim = 1))
with torch.no_grad():
    output = model1(img)
print("Prediction: ", F.softmax(output, dim = 1))
print(label)
imgshow = img.squeeze(0).permute(1, 2, 0).numpy()
plt.imshow(imgshow, cmap = 'gray')
plt.show()

# GradCam

layers1 = [2, 8, 16, 24]
layers2 = [3, 11, 16]
gradcams1 = []
gradcams2 = []
for i in range(4):
    target_layer = model1.features[layers1[i]]
    cam = GradCAM(model=model1, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img)[0]
    base_img = img[0].cpu().numpy()
    base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
    rgb_img = np.stack([base_img]*3, axis=-1)[0]
    gradcams1.append(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))
for i in range(3):
    target_layer = model2.features[layers2[i]]
    cam = GradCAM(model=model2, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img)[0]
    base_img = img[0].cpu().numpy()
    base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
    rgb_img = np.stack([base_img]*3, axis=-1)[0]
    gradcams2.append(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))
gradcams2.append(imgshow)
fig, axes = plt.subplots(2, 4, figsize=(12, 5))

for i in range(4):
    axes[0, i].imshow(gradcams1[i], cmap = "viridis")
    axes[0, i].set_title(f"Block {i}")
    axes[0, i].axis("off")
for i in range(3):
    axes[1, i].imshow(gradcams2[i], cmap = "viridis")
    axes[1, i].set_title(f"Block {i}")
    axes[1, i].axis("off")
axes[1, 3].imshow(imgshow, cmap = 'gray')
axes[1, 3].set_title("Original")
axes[1, 3].axis("off")
plt.tight_layout()
plt.show()


# guided backprop
class GuidedReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)

    def backward(self, grad_output):
        # Nur positive Gradienten durchlassen
        return torch.clamp(grad_output, min=0.0)

def replace_relu_with_guided(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedReLU())
        else:
            replace_relu_with_guided(module)

def guided_backprop_layer(model, image, target_class, target_layer):
    model.zero_grad()
    image.requires_grad = True

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output
    # das hier ist noch quatsch
    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Hook setzen
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    # Forward + Backward
    output = model(image)
    loss = output[0, target_class]
    loss.backward()

    # Hooks entfernen
    handle_f.remove()
    handle_b.remove()

    return activations["value"].detach(), gradients["value"].detach()


replace_relu_with_guided(test_model)
target_layer = test_model.features[8]
image = img.clone().requires_grad_(True)

act, grad = guided_backprop_layer(test_model, image, target_class=5, target_layer=target_layer)

# Visualisierung (z. B. stärkste Feature-Map)
index = torch.argmax(act[0].mean(dim=(1, 2)))
heatmap = grad[0, index].cpu()

heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
plt.imshow(heatmap, cmap="viridis")
plt.axis("off")
plt.show()
imagesgb = []
for i in range (5):
    act, grad = guided_backprop_layer(test_model, images[i], target_class=5, target_layer=target_layer)
    index = torch.argmax(act[0].mean(dim=(1, 2)))
    heatmap = grad[0, index].cpu()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    imagesgb.append(heatmap)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(5):
    axes[0, i].imshow(images[i].squeeze(0).permute(1, 2, 0).detach().numpy(), cmap = "gray")  # falls PyTorch: C,H,W → H,W,C
    axes[0, i].axis("off")

for i in range(5):
    axes[1, i].imshow(imagesgb[i], cmap = "viridis")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()
"""

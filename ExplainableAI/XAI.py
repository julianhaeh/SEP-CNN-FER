import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
import random
from Data.clsOurDataset import OurDataset
from ModelArchitectures.clsCustomVGG13Reduced import CustomVGG13Reduced
from ModelArchitectures.clsDownsizedCustomVGG13Reduced import DownsizedCustomVGG13Reduced
from ModelArchitectures.clsGAP_4_64_96_128_196 import GAP_4_64_96_128_196
from ModelArchitectures.clsReducedClassifierCustomVGG13Reduced import ReducedClassifierCustomVGG13Reduced
from ExplainableAI.GradCAM import OurGradCAM
from ExplainableAI.SaliencyMaps import OurSaliencyMaps
from ExplainableAI.FeatureMaps import OurFeatureMaps
def show_image (image_to_show):
    # converts tensor to numpy image and shows it
    imgshow = image_to_show.squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(imgshow, cmap = 'gray')
    plt.show()
def get_random_images(label = None, amount = 10):
    # returns list of x images of a given or any label
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

dataset = OurDataset(split='test')

Original = CustomVGG13Reduced()
weights1 = torch.load("Experiments/Models/VGG13_Weighted_CE_Acc_72.30_Model.pth", map_location=torch.device('cpu'))
Original.load_state_dict(weights1)

DownsizedVGG13 = DownsizedCustomVGG13Reduced()
weights2 = torch.load("Experiments/Models/CustomVGG13_Downsized_Acc_72.51_Model.pth", map_location=torch.device('cpu'))
DownsizedVGG13.load_state_dict(weights2)
DownsizedVGG13.eval()

ReducedClassifierVGG13 = ReducedClassifierCustomVGG13Reduced() # final model
weights4 = torch.load("Experiments/Models/ReducedClassifier_Weighted_CE_Weighted_Acc_72.84_Model.pth", map_location=torch.device('cpu'))
ReducedClassifierVGG13.load_state_dict(weights4)
ReducedClassifierVGG13.eval()

GAP = GAP_4_64_96_128_196()
weights4 = torch.load("Experiments/Models/GAP_4_64_96_128_196_Weighted_CE_Weighted_Acc_72.10_Model.pth", map_location=torch.device('cpu'))
GAP.load_state_dict(weights4)
GAP.eval()



test_model = ReducedClassifierVGG13
imgs, lbls = get_random_images(label = 3, amount=20)

with torch.no_grad():
    output = test_model(imgs[0])
    probabilites = F.softmax(output, dim=1)[0]
    print("Prediction:", probabilites.tolist())
    prediction = probabilites.argmax().item()
    print("Predicted class:", prediction)
    print("True label: ", lbls[0])

show_image(imgs[0])

GradCAM = OurGradCAM(test_model)
GradCAM.kGradCAMs(imgs, 7)
GradCAM.GradCAM(imgs[0], 7)

FeatureMaps = OurFeatureMaps(test_model)
FeatureMaps.FeatureMaps(imgs[0], 4)
FeatureMaps.get_top_feature_maps(imgs[0], 7, metric = "max")
FeatureMaps.get_top_feature_maps(imgs[0], 7, metric = "mean")

TestSaliencyMaps = OurSaliencyMaps(test_model)
TestSaliencyMaps.SaliencyMap(imgs[0])
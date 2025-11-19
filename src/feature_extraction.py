import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch
from dataset import get_balanced_cifar10
import numpy as np
from sklearn.decomposition import PCA
     
train_balanced, test_balanced = get_balanced_cifar10()

# transform to 224 x 224 and normalize
def get_resnet_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

# get the resnet18 pretrained
def build_resnet_feature_extractor():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity() #remove last layer
    model.eval() #not training
    return model

def extract_resnet_features(dataset, model, transform):
    with torch.no_grad(): #not training
        features_list = []
        labels_list = []
        for index in range(len(dataset)):
            image, label = dataset[index] #get image and label
            image = transform(image) #transform
            image = image.unsqueeze(0) #add batch dimension
            output = model(image)  #pass to model
            output = output.squeeze(0)  #remove batch dimension
            features = output.numpy() #convert to numpy
            features_list.append(features)
            labels_list.append(label)
        X = np.vstack(features_list)  
        y = np.array(labels_list)
    return X,y

transform = get_resnet_transform()
model = build_resnet_feature_extractor()

# X -> (5000,512)
# y -> (5000)

X_train_512, y_train = extract_resnet_features(train_balanced, model, transform)
X_test_512, y_test = extract_resnet_features(test_balanced, model, transform)

#2d numpy array, reduce 512 features to 50 using PCA
pca = PCA(n_components=50)
X_train_50 = pca.fit_transform(X_train_512)
X_test_50 = pca.transform(X_test_512)
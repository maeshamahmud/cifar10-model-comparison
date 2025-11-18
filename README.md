# cifar10-model-comparison
Implementation and evaluation of feature-based ML models and convolutional networks using CIFAR-10.

STEP 1: Prepping Data
1. get 500 training and 100 test images for each 10 classes
2. convert to tensors from PIL
3. convert to low dimension vectors
4. transform to 224x224x3 and then use resnet18 but remove last layer of resnet18
5. use PCA to further remove size of feature vector

Step 2: Naive Bayes

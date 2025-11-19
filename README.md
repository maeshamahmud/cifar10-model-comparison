# cifar10-model-comparison
Implementation and evaluation of feature-based ML models and convolutional networks using CIFAR-10.

STEP 1: Prepping Data
1. get 500 training and 100 test images for each 10 classes
2. convert to tensors from PIL
3. convert to low dimension vectors
4. transform to 224x224x3 and then use resnet18 but remove last layer of resnet18
5. use PCA to further remove size of feature vector from 512 to 50

Step 2: Naive Bayes
  2.1 From Scratch
  1. train for each class the mean and variance of each feature and prior probability of that     class
  2. calculate the gaussian log likelihood
  3. make predictions
  2.2 using scikit-learn

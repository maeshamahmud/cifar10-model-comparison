# cifar10-model-comparison
Implementation and evaluation of feature-based ML models and convolutional networks using CIFAR-10.


This project follows the requirements to:
- Preprocess CIFAR-10 into balanced subsets  
- Extract deep features using a pre-trained convolutional neural network  
- Train multiple models using both manual implementations (NumPy only) and scikit-learn versions  
- Compare performance across all models  
- Report accuracy, precision, recall, F1-score, and confusion matrices  

Project Structure
src/
├── dataset.py
├── feature_extraction.py
├── naive_bayes_manual.py
├── naive_bayes_sklearn.py
├── decision_tree_manual.py
├── evaluation.py
└── main.py


Prepping Data:
1. get 500 training and 100 test images for each 10 classes
2. convert to tensors from PIL
3. convert to low dimension vectors
4. transform to 224x224x3 and then use resnet18 but remove last layer of resnet18
5. use PCA to further remove size of feature vector from 512 to 50

Files:
- dataset.py:
this file was used to create a function that is able to take a desired number of images from a dataset, in this case, it was 500 images for training for each class and 100 test images for each class. The images were taken from cifar10 which was first transformed from PIL to tensor. The function returns the training and testing subset.

- feature_extraction.py: in this file, I began for aspire, creating a function that will resize the images to 224 x 224 and then normalize them. I then load the pre-trained resnet-18 for which I removed the last layer and put it in evaluation mode since we're not training it. I then wrap it around, no gradient and take the image and label. I add a batch dimension to the image and then pass it through the model and then remove the batch dimension and then converted to numpy. I'm left with an X and a Y. And then prepare the data by having my training and testing subsets which I transform and using the function that I've created I get an X and a Y for each training and testing subset. The X has the images and the feature while the wire I have all the labels. I run the X's through PCA to reduce the features from 512 to 50 and I return the X and the way of the training and testing subsets.

- naive_bayes_manual.py:
- naive_bayes_sklearn.py:
- decision_tree_manual.py:
- evaluation.py:
- main.py:

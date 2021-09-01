# CSS490_Final_Project
<h2>Overview </h2>

We will have a goal of analyzing two well-known face recognition datasets ([VGG Face](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/) and [UTKFace](https://susanqq.github.io/UTKFace/)) and explore if there are biases in two datasets by training two identical models (ResNet) and compare the accuracy of each model. We added [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet) for an artificial dataset.


![](Images/UTK.png)
![](Images/VGG.png)
![](Images/TinyImageNet.png)
![](Images/Methodology.png)
![](Images/Architecture.png)
![](Images/Accuracy.png)
![](Images/EncounteredIssue.png) 

<h2>Execution</h2>
We recommend use coLab because of high usage of GPU
  <li>1. Clone the repository</li>
  <li>2. Open <code>main_colab.ipynb</code></li>
  <li>3. Uncommnet lines from Import statements</li>

```py
from google.colab import drive  
drive.mount('/content/drive')
!pwd
!ls drive/MyDrive/CSS490/
!mkdir ./datasets
!cp drive/MyDrive/CSS490/modified_datasets.tar.gz ./datasets
!mkdir ./util
!cp -r drive/MyDrive/CSS490/util/ ./
!tar -xf ./datasets/modified_datasets.tar.gz --directory=./datasets/
!python --version  
```
  <li>3. Click Runtime at the top and select <code>Run all</code></li>










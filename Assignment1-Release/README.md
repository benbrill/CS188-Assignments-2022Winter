## Assignment 1
Welcome to CS188 Computer Vision course!

The goals of the first assignment are as follows:
* Gain experience using notebooks on Google Colab
* Understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
* Gain experience preparing and loading data
* Implement and apply linear/logistic/softmax regression for classification
* Implement and apply a k-Nearest Neighbor (kNN) classifier

This assignment is due on **Sunday, Jan 23**

### Q1: Data Preparation (20 pts)
For Q1, you need to create TinyPlaces from MiniPlaces https://github.com/CSAILVision/miniplaces. 

Detailed instructions are as below:
* Download the miniplaces zip:
https://drive.google.com/file/d/16GYHdSWS3iMYwMPv5FpeDZN2rH7PR0F2/view. The “images” folder contains three folders: train, val and test.
* You need to create **tinyplaces-train** and **tinyplaces-val** from the miniplaces dataset. 

  * You should have a python file called **“tinyplaces_creator.py”** to create the TinyPlaces dataset. 
Select the images from the target subcategories in the file categories_tinyplaces.txt, and create CIFAR-like training and validation files for TinyPlaces.
  * All images should be resized to 32 * 32.
  * There are in total 20 subcategories. The first 10 subcategories are indoor scenes. The last 10 subcategories are outdoor scenes. 
  * We will first focus on binary classification on indoor and outdoor scenes. We give the images in the 10 indoor subcategories the label “0”, and the images in the 10 outdoor categories the label “1”.
  * For training data, we take the first 500 images of each subcategory in the train image folder. The ids we want are specified in https://github.com/CSAILVision/miniplaces/tree/master/data/train.txt. The training set contains 10,000 images in total.
  * For validation data, we take the first 50 images of each subcategory in the validation image folder. The ids we want are specified in https://github.com/CSAILVision/miniplaces/tree/master/data/val.txt. The validation set contains 1,000 images in total.
  * Create two CIFAR-like files: tinyplaces-train and tinyplaces-val.
CIFAR is a large dataset for image classification. Please refer to https://www.cs.toronto.edu/~kriz/cifar.html for details. Here, we create TinyPlaces as a dataset like CIFAR for python. Specifically, we need to create a dictionary with two keys: “data” and “label”. 
For tinyplaces-train, the key “data” contains a (10000, 3072) numpy array representing the image data.
The key “label ” contains a (10000,) numpy array representing the label data. 
The image data and label data should be randomly shuffled using numpy’s random.shuffle function.
For how to create CIFAR like dataset, you may refer to https://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10 (However, we use pickle to save the data, and we have two keys "data" and "label" instead of putting them together). 
  * tinyplaces-train and tinyplaces-val should be dumped using pickle under a folder named “data” in the assignment directory.

* You need to create **tinyplaces-train-multiclass** and **tinyplaces-val-multiclass** 
  The procedure is the same as before, except that we now focus on **multi-class classification where we have 20 categories**. Modify your tinyplaces_creator.py. Instead of having binary labels ("0" and "1"), we now have 20 labels (0~19) for the 20 categories. 
  * Save tinyplaces-train-multiclass and tinyplaces-val-multiclass under a folder named “data” in the assignment directory.
 
 ### Q2: Data Loading (5 pts)
 For Q2, you will implement a function that loads the data for visualiztion and model training.
 
 The notebook **data_loader.ipynb** will walk you through loading data for visualization. You are required to write code on **cs188/data.py**.
 
 ### Q3: Linear / Logistic /Softmax Regression (40 pts)
 The notebook **regression.ipynb** will walk you through implementing linear / logistic /softmax regression. Your implementation will go to **regression.py**.
  
 ### Q4: KNN Classification (35 pts)
The notebook **knn.ipynb** will walk you through implementing a data loader for pytorch. Your implementation will go to **knn.py**.

## Steps
* Create TinyPlaces locally
* Download the Github repo using fork/clone
* Unzip the repo and open the Colab file from the Drive

 Once you unzip the downloaded content, please upload the folder and the data to your Google Drive. Then, open each * .ipynb notebook file with Google Colab by right-clicking the * .ipynb file. No installation or setup is required! 

* Open your corresponding * .py from Google Colab and work on the assignment
  Next, we recommend editing your * .py file on Google Colab, set the ipython notebook and the code side by side. Work through the notebook, executing cells and implementing the codes in the * .py file as indicated. You can save your work, both * .ipynb and * .py, in Google Drive (click “File” -> “Save”) and resume later if you don’t want to complete it all at once.

While working on the assignment, keep the following in mind:
* The notebook and the python file have clearly marked blocks where you are expected to write code. 
* Do not write or modify any code outside of these blocks.
* Do not add or delete cells from the notebook. You may add new cells to perform scratch computations, but you should delete them before submitting your work.
* Run all cells, and do not clear out the outputs, before submitting. You will only get credit for code that has been run.


## Submission
Please click "Share" the assignment folder in your Google Drive and copy the link to submit.

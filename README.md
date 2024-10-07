Pulmonary Cancer Detection Using Deep Learning

This project aims to detect and classify lung nodules in CT scan images to aid in the early diagnosis of pulmonary cancer. The project utilizes a Convolutional Neural Network (CNN) for feature extraction and classification of the images. The implementation is done using Python, TensorFlow, and Keras within a Jupyter notebook running in Google Colab. Additionally, a Flask-based website is provided where users can upload CT scan images and receive predictions.

^--Table of Contents--^

* Overview
* Dataset
* Requirements
* Installation
* Model
   * Simple CNN Model
   * CNN Model with Dropout Layers
   * CNN Model with Data Augmentation Layers
   * CNN Model with Filters in All Layers
* How to Run
* Results
* Website
* Contributing
* Contact

^--Overview--^
Pulmonary cancer is one of the leading causes of cancer-related deaths worldwide. Early detection is crucial for improving survival rates. This project focuses on building a deep learning model to classify lung nodules as cancerous or non-cancerous, based on CT scan images.

The project includes the following components:
--Data loading and preprocessing
--Model architecture using Convolutional Neural Networks (CNNs)
--Model training and evaluation
--A website interface for users to upload CT scan images for predictions


^--Dataset--^
The dataset used for this project consists of CT scan images and can be accessed (https://drive.google.com/drive/folders/1vcsE0Ue916DkwpVl2ZkST3TeBFkjMgtO?usp=sharing). It contains labeled data of lung nodules, which are used for training and evaluating the models.

^--Requirements--^
The following Python libraries are required to run this project:
--TensorFlow
--Keras
--scikit-learn
--matplotlib
--pandas
--numpy
You can install these dependencies by running:
pip install tensorflow keras scikit-learn matplotlib pandas numpy

^--Installation--^
To run the project locally or on Google Colab, follow these steps:

--Clone the repository:
        git clone https://github.com/your-username/pulmonary-cancer-detection.git
        cd pulmonary-cancer-detection
--Install the required libraries:
        pip install -r requirements.txt
--To run in Google Colab, upload the notebook file (pulmonary_cancer_detection_using_deep_learning.ipynb) and ensure all necessary dependencies are installed.

^--Model--^
This project explores multiple CNN architectures for lung cancer detection. The models include:

** Simple CNN Model
     A straightforward convolutional neural network with basic layers (convolutional, pooling, and dense).
     Suitable for baseline comparisons.
** CNN Model with Dropout Layers
     Includes dropout layers to prevent overfitting by randomly setting units to zero during training.
     Helps improve model generalization.
** CNN Model with Data Augmentation Layers
     Implements data augmentation (e.g., rotation, zoom, shift) to artificially expand the training dataset.
     Reduces overfitting and enhances model performance by introducing variability.
** CNN Model with Filters in All Layers
     This model uses filters in all layers to capture more complex features.
     The architecture includes multiple convolutional layers with filters applied across all layers, leading to better feature extraction.
     
^--How to Run--^
1.Open the Jupyter notebook (pulmonary_cancer_detection_using_deep_learning.ipynb) in your local environment or in Google Colab.
2.Follow the steps in the notebook for data loading, preprocessing, model training, and evaluation.
3.You can modify hyperparameters, models, and other settings within the notebook to experiment with the different CNN architectures.
4.To test the website, refer to the provided Flask files. The website allows users to upload CT scan images and get predictions on the cancer type.

^--Results--^
The results from each model, including accuracy, loss, and confusion matrix, are documented in the notebook. Each model has been evaluated, and the final CNN model with data augmentation and filters showed the highest accuracy in classifying lung nodules.

^--Website--^
A website has been developed using Flask, where users can upload their CT scan images and receive predictions based on the trained model. You can check out the Flask application files in this repository for more details.

^--Contributing--^
Contributions are welcome! Feel free to submit issues or pull requests to help improve the project. Please make sure your code adheres to the existing style guidelines and is well-documented.

^--Contact--^
If you have any questions or doubts, feel free to contact me via email at vankayalapatijaashvitha@gmail.com.

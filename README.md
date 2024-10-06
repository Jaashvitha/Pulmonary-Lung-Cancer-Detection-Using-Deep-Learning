Lung Cancer Detection Web Application

This repository contains a Flask-based web application that uses a CNN model built with TensorFlow to detect and classify lung nodules from CT scan images. The model is trained to predict different types of lung cancer with the highest algorithm accuracy. The web application allows users to upload CT scan images and returns predictions based on the trained model.

Features-
* Upload CT scan images through the web interface.
* Predicts the type of lung cancer using a CNN model.
* Real-time analysis with the highest accuracy based on model performance.
* Simple and user-friendly interface powered by Flask.
* Ngrok integration for easy local-to-web access and testing.
  
Prerequisites-

Before running the project, ensure you have the following installed:
* Python 3.x
* TensorFlow
* Flask
* OpenCV
* Ngrok
* Numpy
* Matplotlib
  
You can install the necessary libraries by running:
pip install -r requirements.txt
Getting Started
Clone the repository:
git clone https://github.com/yourusername/lung-cancer-detection.git
cd lung-cancer-detection
Train the model (Optional):

If you wish to retrain the model, ensure your dataset is in the correct directory and modify the paths in the script accordingly.
python train_model.py
Run the Flask web app:
python app.py
Use Ngrok for external access:
If testing externally, open a tunnel to the Flask application using Ngrok:
ngrok http 5000
Copy the public URL from Ngrok to access the web app.

Project Structure
plaintext
Copy code
lung-cancer-detection/
│
├── models/                # CNN model and training scripts
├── static/                # CSS and images for the web app
├── templates/             # HTML templates for the web app
├── app.py                 # Flask application file
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
└── README.md              # Project overview

How It Works-
Upload Image: Users upload CT scan images via the web interface.
Model Prediction: The CNN-based model processes the image and predicts the type of cancer (adenocarcinoma, large-cell carcinoma, squamous cell carcinoma, or normal).
Display Results: The web app displays the prediction and the model's accuracy.
Model Details
The CNN model was developed using TensorFlow and trained on a dataset of labeled CT scan images. The model uses convolutional layers to extract features and fully connected layers for classification.

Future Improvements-
Enhance model accuracy through data augmentation and hyperparameter tuning.
Add support for additional types of cancer.
Deploy the application using a production-ready server.
License
This project is licensed under the MIT License.

Contact
For any inquiries or suggestions, feel free to contact me at [vankayalapatijaashvitha@gmail.com].

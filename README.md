Facial Emotion Recognition Using CNN–LSTM By Abdul Wasey

Abstract
Facial Emotion Recognition (FER) is an important research area in computer vision and affective computing. This project presents a deep learning–based facial emotion recognition system using a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture. The model is trained on the FER2013 dataset and optimized using the Adam optimizer. Experimental results demonstrate that the proposed approach effectively classifies facial expressions into seven emotion categories with competitive accuracy.

1. Introduction
Human emotions play a crucial role in communication and decision-making. Automatic facial emotion recognition has applications in human–computer interaction, healthcare, surveillance, education, and entertainment. However, recognizing emotions from facial images is challenging due to variations in lighting, pose, occlusion, and subtle differences between expressions. This project aims to design and implement a robust facial emotion recognition system using deep learning techniques.

3. Objectives
The main objectives of this project are:
• To study facial emotion recognition using deep learning.
• To implement a CNN-based feature extractor for facial images.
• To integrate an LSTM network to learn feature dependencies.
• To train and evaluate the model on the FER2013 dataset.
• To analyze model performance using accuracy and confusion matrix.

5. Dataset Description
The FER2013 dataset is a widely used benchmark dataset for facial emotion recognition. It consists of 35,887 grayscale facial images of size 48×48 pixels. The dataset is divided into training and test sets and includes seven emotion classes:
Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The dataset is challenging due to low image resolution, noise, and class imbalance.

4. Methodology
The proposed system follows a multi-stage pipeline consisting of data preprocessing, feature extraction, sequence learning, and classification.

4.1 Data Preprocessing and Augmentation
All images are resized to 48×48 pixels and normalized to the range [0,1]. Data augmentation techniques such as rotation, width and height shifting, zooming, shearing, and horizontal flipping are applied to increase dataset diversity and reduce overfitting.

4.2 CNN Feature Extraction
The Convolutional Neural Network (CNN) is used to extract spatial features from facial images. Multiple convolutional layers with ReLU activation functions are employed, followed by max pooling and batch normalization layers. Dropout is applied to improve generalization.

4.3 LSTM for Feature Dependency Learning
After CNN-based feature extraction, the features are reshaped and passed to an LSTM layer. The LSTM network captures dependencies between extracted features, enhancing the model’s ability to recognize subtle emotional patterns.

4.4 Classification and Optimization
The model is trained using the Adam optimizer, which adaptively adjusts the learning rate for faster and more stable convergence. Categorical cross-entropy is used as the loss function.

6. Model Architecture
The proposed CNN–LSTM architecture consists of:
• Three convolutional blocks (Conv2D + Batch Normalization + MaxPooling + Dropout)
• A Flatten layer
• A Reshape layer to prepare features for LSTM
• One LSTM layer
• A Dense Softmax output layer with seven neurons

This hybrid architecture combines spatial and sequential learning capabilities.

6. Training and Evaluation
The model is trained for multiple epochs with a batch size of 64. Validation data is used to monitor performance and prevent overfitting. Model performance is evaluated using accuracy and confusion matrix on the test dataset.

8. Results and Discussion
The CNN–LSTM model achieved improved accuracy compared to a baseline CNN model. Data augmentation and the Adam optimizer contributed to stable training and better generalization. The confusion matrix shows that emotions such as Happy and Surprise are recognized with higher accuracy, while Fear and Disgust remain more challenging due to visual similarities.

10. Conclusion
This project successfully implemented a facial emotion recognition system using a CNN–LSTM hybrid model. The use of the Adam optimizer and data augmentation improved training stability and accuracy. Experimental results confirm the effectiveness of deep learning techniques for emotion recognition tasks.


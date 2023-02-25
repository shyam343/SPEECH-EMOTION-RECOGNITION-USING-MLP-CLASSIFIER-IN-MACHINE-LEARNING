# SPEECH-EMOTION-RECOGNITION-USING-MLP-CLASSIFIER-IN-MACHINE-LEARNING.


![Speech-Emotion-Recognition-System](https://user-images.githubusercontent.com/84222697/221345257-de8e65d0-e9bb-46d0-b962-daa7d947a631.png)

## Overview:-

Speech Emotion Recognition (SER) is a field of study that focuses on developing algorithms and models to recognize emotions from speech signals. The use of Machine Learning (ML) techniques has become increasingly popular in SER due to their ability to extract useful features from speech signals and classify emotions accurately. In this Project, MLP (Multi-Layer Perceptron) is a type of artificial neural network commonly used in ML for classification tasks.

The "Speech Emotion Recognition Using MLP Classifier in Machine Learning" presents an overview of the SER system using MLP classifiers. The proposed system includes preprocessing of speech signals, feature extraction, and classification using MLP classifiers. The study suggests that the proposed MLP-based classification system can achieve high accuracy in recognizing emotions from speech signals.

The paper first describes the preprocessing of speech signals, which includes several steps such as noise removal, signal normalization, and segmentation. Then, feature extraction methods such as Mel-frequency cepstral coefficients (MFCCs) and Linear Predictive Coding (LPC) are used to extract relevant features from the preprocessed speech signals. These features are then used as inputs for the MLP classifier.

The MLP classifier is trained using a labeled dataset that includes speech signals and their corresponding emotional labels. The project describes the training process and the parameters used to optimize the MLP classifier's performance. The MLP classifier's output is the predicted emotion label, which is compared to the actual emotion label to measure the system's accuracy.

The project concludes that MLP classifiers can achieve high accuracy in recognizing emotions from speech signals. However, the accuracy of the system depends on several factors such as the quality of the speech signal, the feature extraction methods used, and the size and quality of the labeled dataset used for training. The study suggests that future research should focus on improving the accuracy of MLP classifiers by optimizing the feature extraction methods and using larger and more diverse labeled datasets for training.

##  Dependencies Libraries:-
1. Programming language: The system can be implemented using programming languages such as Python.
2. Libraries for speech processing: There are several libraries available for speech processing, such as PyAudio, librosa,Soundfile,numpy,sklearn,pandas, and SpeechRecognition. These libraries are used for tasks such as recording, preprocessing, and feature extraction of speech signals.
3. Libraries for machine learning: The system requires machine learning libraries for building and training the MLP classifier. Popular machine learning libraries include scikit-learn, TensorFlow, and Keras.

4. Dataset: The system requires a labeled dataset of speech signals and their corresponding emotional labels for training and testing the MLP classifier.

5. Development environment: Depending on the programming language used, a suitable development environment, such as Jupyter Notebook,or Eclipse, may be required to develop and run the system.

6. Hardware: The system requires a computer with suitable processing power and memory capacity to handle the speech processing and machine learning tasks. High-end CPUs or GPUs may be required for training large MLP models.

## Project Details:
The models which were discussed in the repository are MLP,SVM,Decision Tree,CNN,Random Forest and neural networks of mlp and CNN with different architectures.
+ utilities.py - Contains extraction of features,loading dataset functions
- loading_data.py - Contains dataset loading,splitting data
* mlp_classifier_for_SER.py - Contains mlp model code
+ SER_using_ML_algorithms.py - Contains SVM,randomforest,Decision tree Models.
- Speech_Emotion_Recognition_using_CNN.ipynb - Consists of CNN-1d model

## NOTE : Remaining .ipynb files were same as above files but shared from google colab.

# Dataset Source - Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
In this project, I use [RAVDESS](https://zenodo.org/record/1188976#.Y_nGH3ZBy3B) dataset to train.
![speechs](https://user-images.githubusercontent.com/84222697/221347443-846586c1-81d3-47fb-8c7c-978080adbf84.png)

You can find this dataset in kaggle or click on below link.
[Link](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

In This Dataset 2452 audio files, with 12 male speakers and 12 Female speakers, the lexical features (vocabulary) of the utterances are kept constant by speaking only 2 statements of equal lengths in 8 different emotions by all speakers. This dataset was chosen because it consists of speech and song files classified by 247 untrained Americans to eight different emotions at two intensity levels: Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprise, along with a baseline of Neutral for each actor.


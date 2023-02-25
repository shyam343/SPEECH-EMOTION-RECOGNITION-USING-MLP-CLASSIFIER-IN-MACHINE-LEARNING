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

## Data preprocessing :
The heart of this project lies in preprocessing audio files. If you are able to do it . 70 % of project is already done. We take benefit of two packages which makes our task easier. - LibROSA - for processing and extracting features from the audio file. - soundfile - to read and write audio files in the storage.

The main story in preprocessing audio files is to extract features from them.

Features supported:

+ MFCC (mfcc)
- Chroma (chroma)
/ MEL Spectrogram Frequency (mel)
* Contrast (contrast)
- Tonnetz (tonnetz)
In this project, code related to preprocessing the dataset is written in two functions.

- load_data()
+ extract_features()


load_data() is used to traverse every file in a directory and we extract features from them and we prepare input and output data for mapping and feed to machine learning algorithms. and finally, we split the dataset into 80% training and 20% testing.

```ruby
def load_data(test_size=0.2):
  X, y = [], []
  try :
    for file in glob.glob("/content/drive/My Drive/wav/Actor_*/*.wav"):
          # get the base name of the audio file
        basename = os.path.basename(file)
        print(basename)
          # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
          # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
              continue
          # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
          # add to data
        X.append(features)
        y.append(emotion)
  except :
       pass
    # split the data to training and testing and return it
  return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
  ```
  Below is the code snippet to extract features from each file.
  ```ruby
  
def extract_feature(file_name, **kwargs):
  """
  Extract feature from audio file `file_name`
      Features supported:
          - MFCC (mfcc)
          - Chroma (chroma)
          - MEL Spectrogram Frequency (mel)
          - Contrast (contrast)
          - Tonnetz (tonnetz)
      e.g:
      `features = extract_feature(path, mel=True, mfcc=True)`
  """
  mfcc = kwargs.get("mfcc")
  chroma = kwargs.get("chroma")
  mel = kwargs.get("mel")
  contrast = kwargs.get("contrast")
  tonnetz = kwargs.get("tonnetz")
  with soundfile.SoundFile(file_name) as sound_file:
      X = sound_file.read(dtype="float32")
      sample_rate = sound_file.samplerate
      if chroma or contrast:
          stft = np.abs(librosa.stft(X))
      result = np.array([])
      if mfcc:
          mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
          result = np.hstack((result, mfccs))
      if chroma:
          chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
          result = np.hstack((result, chroma))
      if mel:
          mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
          result = np.hstack((result, mel))
      if contrast:
          contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
          result = np.hstack((result, contrast))
      if tonnetz:
          tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
          result = np.hstack((result, tonnetz))
  return result
 ```
 Remaining drive further into the project ...
 
## Training and Analysis:
Traditional Machine Learning Models

Performs different traditional algorithms such as -Decision Tree, SVM, Random forest .

Project Code:-

[Speech Emotion Recognition using CNN](https://github.com/shyam343/SPEECH-EMOTION-RECOGNITION-USING-MLP-CLASSIFIER-IN-MACHINE-LEARNING/blob/main/Speech%20Emotion%20Recognition%20using%20CNN.ipynb)

##  Conclusion and Analysis :

+  Speech Emotion Recognition using MLP Classifier in Machine Learning requires careful attention to the quality of the labeled dataset, preprocessing methods, feature extraction techniques, and optimization of hyperparameters.
-  The demand for accurate and reliable speech emotion recognition systems is likely to continue to grow in the future, with applications in emotional health monitoring, voice-based personal assistants, and human-robot interactions.

*  Advanced techniques such as deep neural networks and combining multiple classifiers can improve the performance of Speech Emotion Recognition systems and should be explored by researchers and practitioners.

#  The Accuracy of Our Project is 81 %.

##### Hope You all Get Idea How The SPEECH-EMOTION-RECOGNITION-USING-MLP-CLASSIFIER-IN-MACHINE-LEARNING Working.
# THANKS YOU!

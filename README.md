# SPEECH-EMOTION-RECOGNITION-USING-MLP-CLASSIFIER-IN-MACHINE-LEARNING.


![Speech-Emotion-Recognition-System](https://user-images.githubusercontent.com/84222697/221345257-de8e65d0-e9bb-46d0-b962-daa7d947a631.png)

## Overview:-

SPEECH-EMOTION-RECOGNITION-USING-MLP-CLASSIFIER-IN-MACHINE-LEARNING is a system that uses machine learning techniques to recognize the emotional state of a speaker based on their speech. The system utilizes a Multi-Layer Perceptron (MLP) classifier to classify the emotional state of the speaker into different categories such as happy, sad, angry, etc. The system is trained on a dataset of speech samples labeled with corresponding emotions to learn patterns and features that are associated with each emotional state. Once trained, the system can accurately classify the emotional state of new speech samples, making it useful for applications such as call center monitoring, speech therapy, and psychological research.

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
 
 ## Audio files:
 Tested out the audio files by plotting out the waveform and a spectrogram to see the sample audio files.
Waveform.
![1 Wave pic](https://user-images.githubusercontent.com/84222697/221512776-51188a53-8707-4ae4-b67e-ba5f6324b037.png)

######  Spectrogram
![2spectrom](https://user-images.githubusercontent.com/84222697/221512802-c46d575b-f889-4aeb-a8d2-00e45ccaf7e8.png)
##  Feature Extraction From Audio File.
The next step involves extracting the features from the audio files which will help our model learn between these audio files. For feature extraction we make use of the [Librosa](https://librosa.org/doc/latest/index.html) library in python which is one of the libraries used for audio analysis.
 ![feature extraction](https://user-images.githubusercontent.com/84222697/221514130-cea2d013-41cb-4797-b707-48cabea018e0.png)
+ Here there are some things to note. While extracting the features, all the audio files have been timed for 3 seconds to get equal number of features.
- The sampling rate of each file is doubled keeping sampling frequency constant to get more features which will help classify the audio file when the size of dataset is small.


#####    The extracted features looks as follows:-
![feature2](https://user-images.githubusercontent.com/84222697/221514661-2f7ebc38-667e-4c50-9714-421430c907f8.png)

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

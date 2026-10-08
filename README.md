# CodeAlpha Speech Emotion Recognition

Recognizing human emotions from speech audio using MFCC feature extraction and machine learning.
Task 2 of the CodeAlpha Machine Learning Internship.

## Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song): 1440 audio samples from 24 actors, covering 8 emotions (angry, calm, disgust, fearful, happy, neutral, sad, surprised).
Downloaded from Kaggle.

## Workflow
1. Audio exploration: waveform and spectrogram visualization
2. Feature extraction: 40 MFCC (Mel-Frequency Cepstral Coefficients) per sample, averaged over time into a fixed-size vector
3. Label encoding and stratified train/test split (80/20), feature scaling
4. Models: Logistic Regression, SVM, Random Forest, MLP (Neural Network)
5. Evaluation: Accuracy, Precision, Recall, F1 (weighted), confusion matrix

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.438 | 0.432 | 0.438 | 0.427 |
| SVM | 0.594 | 0.631 | 0.594 | 0.575 |
| Random Forest | 0.649 | 0.653 | 0.649 | 0.637 |
| **MLP (Neural Network)** | **0.733** | **0.732** | **0.733** | **0.730** |

## Key findings
- The MLP neural network performed best, reaching 73.3% accuracy on 8-way emotion classification, well above the 12.5% random-guess baseline.
- Fearful and angry were the easiest emotions to recognize (recall 0.87 and 0.84), with distinct acoustic signatures.
- Neutral and sad were the hardest (recall 0.58 and 0.55), frequently confused with calm, since these emotions share similar low-energy vocal characteristics.
- The approach of converting audio into time-averaged MFCC features is conceptually similar to extracting spectral features from time-series signals in other domains, such as seismology.

## Tools
Python, librosa, scikit-learn, pandas, matplotlib, seaborn

## Notebook
[CodeAlpha_SpeechEmotion.ipynb](CodeAlpha_SpeechEmotion.ipynb)

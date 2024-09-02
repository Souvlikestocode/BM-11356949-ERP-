#!/usr/bin/env python
# coding: utf-8

# # AI ADOPTION IN ASSET MANAGEMENT : SENTIMENT ANALYSIS OF NEWS HEADLINES IN PREDICTING STOCK PRICE MOVEMENT

# ## IMPORTING LIBRARIES REQUIRED FOR THE STUDY  

# In[1]:


## used for data manipulation 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## nlp preprocessing libraries 
import nltk 
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ### Please use !pip install (package) for each packages shown on requirements.txt to ensure the package is installed on the system and code is run efficiently 

# ## DATA EXPLORATION

# ### Please replace the file location when run on local system 

# In[2]:


# dataset with financial headlines 
df = pd.read_csv(r"C:\Users\souvi\Downloads\lh04de0uj42vimh9.csv")


# In[3]:


# display of the first few rows of the dataframe 
df.head()


# In[4]:


# ensuring no company names found are apart from the ones chosen for the study
filter_1 = df[df['companyname'] == 'M&A Rumors and Discussions'].index
df.drop(filter_1, inplace=True)

filter_2 = df[df['companyname'] == 'Visa Inc.'].index
df.drop(filter_2, inplace=True)

df['companyname'].unique()


# In[5]:


# length of dataset 
len(df)


# ### DATASET - STOCK PRICES AND RETURNS

# ### Please replace the file location when run on local system 

# In[6]:


df_returns = pd.read_csv(r"C:\Users\souvi\Downloads\rbkr57yjmsjznlo0.csv")


# In[7]:


# dsplaying the first few rows of the dataframe 
df_returns.head()


# ## WORDCLOUD - to check fo common words in sentences for each company

# In[8]:


from wordcloud import WordCloud

microsoft_headlines = df[df['companyname'] == 'Microsoft Corporation']['headline']
full_text_msft = ' '.join(microsoft_headlines)

wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(full_text_msft)
# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.show()

## no useful keywords


# In[9]:


apple_headlines = df[df['companyname'] == 'Apple Inc.']['headline']
full_text_apple = ' '.join(apple_headlines)

wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(full_text_apple)
# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.show()

## no useful keywords here


# In[10]:


alphabet_headlines = df[df['companyname'] == 'Alphabet Inc.']['headline']
full_text_google = ' '.join(alphabet_headlines)

wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(full_text_google)
# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

## no useful keywords - just name of the company


# In[11]:


netflix_headlines = df[df['companyname'] == 'Netflix, Inc.']['headline']
full_text_netflix = ' '.join(netflix_headlines)

wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(full_text_netflix)
# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.show()

## no useful keywords here - common words such as earnings, guidance which 


# In[12]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

NVIDIA_headlines = df[df['companyname'] == 'NVIDIA Corporation']['headline']
full_text_NVIDIA = ' '.join(NVIDIA_headlines)

wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(full_text_NVIDIA)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

## no useful keywords


# In[13]:


JPMC_headlines = df[df['companyname'] == 'JPMorgan Chase & Co.']['headline']
full_text_JPMC = ' '.join(JPMC_headlines)

wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(full_text_JPMC)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

## main keywords - Fixed Income but does not give us any meaning 


# ## SENTIMENT LABELLING USING FINBERT

# ### FINDING POSITIVE, NEGATIVE, NEUTRAL FOR EACH HEADLINE

# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Loading the FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to('cuda' if torch.cuda.is_available() else 'cpu')

# Function to get sentiment scores
def get_sentiment_scores(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().cpu().numpy().flatten()

# Apply the function to get sentiment scores
df[['negative', 'neutral', 'positive']] = df['headline'].apply(lambda x: pd.Series(get_sentiment_scores(x)))


# ### GENERATING A SENTIMENT LABEL FROM THE HEADLINE

# In[ ]:


# Function to get sentiment scores
def get_sentiment(text):
    # Tokenisng the input text and move to GPU to run quicker
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = model(**inputs)
    # Probablity calculation
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Moving the probablity to CPU and converting to a score on numPy
    return torch.argmax(probs, dim=-1).cpu().item()

# Applying the scores to the headline dataset
df['sentiment_finbert'] = df['headline'].apply(get_sentiment)


# In[16]:


# count of each sentiment label using FinBERT
df['sentiment_finbert'].value_counts()


# ## SENTIMENT LABELLING USING VADER 

# In[17]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# varaible to call the Sentiment Analyser
VADER_analyzer = SentimentIntensityAnalyzer()

# Function to calculate sentiment
def vader_sentiment(text):
    scores = VADER_analyzer.polarity_scores(text)
    
    # categorisation of the sentiment into 3 categories : positive, neutral, or negative
    if scores['compound'] >= 0.05:
        return 2  # Positive
    
    elif scores['compound'] <= -0.05:
        return 0  # Negative
    
    else:
        return 1  # Neutral

# applyinig the sentiment labels into the original dataset on each headline
df['sentiment_VADER'] = df['headline'].apply(vader_sentiment)


# In[18]:


## checking for the value count of each sentiment in the entire dataset
df['sentiment_VADER'].value_counts()


# ## SENTIMENT LABELLING USING L-M Method

# In[19]:


import re
import string
from collections import Counter

# Loading the Loughran-McDonald Master Dictionary
lm_dict = pd.read_csv(r"C:\Users\souvi\Downloads\Loughran-McDonald_MasterDictionary_1993-2023.csv")


# In[20]:


# Extract words for each sentiment category
positive_words = lm_dict[lm_dict['Positive'] > 0]['Word'].str.lower().tolist()
negative_words = lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower().tolist()
uncertainty_words = lm_dict[lm_dict['Uncertainty'] > 0]['Word'].str.lower().tolist()
litigious_words = lm_dict[lm_dict['Litigious'] > 0]['Word'].str.lower().tolist()
strong_modal_words = lm_dict[lm_dict['Strong_Modal'] > 0]['Word'].str.lower().tolist()
weak_modal_words = lm_dict[lm_dict['Weak_Modal'] > 0]['Word'].str.lower().tolist()

# Sentiment scoring function
def score_text(text, positive_words, negative_words, uncertainty_words, litigious_words, strong_modal_words, weak_modal_words):
    words = text.split()
    word_counts = Counter(words)
    
    scores = {
        'positive': sum(word_counts[word] for word in positive_words if word in word_counts),
        'negative': sum(word_counts[word] for word in negative_words if word in word_counts),
        'uncertainty': sum(word_counts[word] for word in uncertainty_words if word in word_counts),
        'litigious': sum(word_counts[word] for word in litigious_words if word in word_counts),
        'strong_modal': sum(word_counts[word] for word in strong_modal_words if word in word_counts),
        'weak_modal': sum(word_counts[word] for word in weak_modal_words if word in word_counts),
    }
    return scores

# Scoring the preprocessed headlines
df['scores'] = df['headline'].apply(
    lambda x: score_text(x, positive_words, negative_words, uncertainty_words, litigious_words, 
                         strong_modal_words, weak_modal_words)
)

# Extract individual scores
df['positive_score'] = df['scores'].apply(lambda x: x['positive'])
df['negative_score'] = df['scores'].apply(lambda x: x['negative'])
df['uncertainty_score'] = df['scores'].apply(lambda x: x['uncertainty'])
df['litigious_score'] = df['scores'].apply(lambda x: x['litigious'])
df['strong_modal_score'] = df['scores'].apply(lambda x: x['strong_modal'])
df['weak_modal_score'] = df['scores'].apply(lambda x: x['weak_modal'])


# In[21]:


# Sentiment classification function
def classify_sentiment(scores):
    net_sentiment = scores['positive'] - scores['negative']
    if net_sentiment > 0:
        return 2
    elif net_sentiment < 0:
        return 0
    else:
        return 1
    
# Classify the sentiment of the headlines
df['sentiment_LM'] = df['scores'].apply(classify_sentiment)


# In[22]:


df['sentiment_LM'].value_counts()


# In[23]:


df['companyname'].value_counts()


# ## DF and DF_RETURNS PREPROCESSING TO ENSURE MERGING IS DONE 

# In[24]:


# renaming date column in df to ensure it matches with df_returns 
df.rename(columns={'announcedate': 'date'}, inplace=True)


# In[25]:


# converting the date column in both datasets to datetime format
df['date'] = pd.to_datetime(df['date'])
df_returns['date'] = pd.to_datetime(df_returns['date'])


# In[26]:


# renaming of columns to ensure merging can be done
df_returns.rename(columns={'COMNAM': 'company_name'}, inplace=True)
df.rename(columns={'companyname': 'company_name'}, inplace=True)


# In[27]:


# modifying the names of the company_names in df_returns to match with df dataset 

df_returns['company_name'] = df_returns['company_name'].apply(
    lambda x: 'JPMorgan Chase & Co.' if x == 'J P MORGAN CHASE & CO' else
              'Microsoft Corporation' if x == 'MICROSOFT CORP' else
              'Alphabet Inc.' if x == 'ALPHABET INC' else
              'Apple Inc.' if x == 'APPLE INC' else
              'NVIDIA Corporation' if x == 'NVIDIA CORP' else
              'Netflix, Inc.' if x == 'NETFLIX INC' else x
)


# ### MERGING THE FINANCIAL NEWS DATASET (DF) AND STOCK PRICE RETURN (DF_RETURNS)

# In[28]:


# Merging the datasets based on common columns to do further analysis 
merged_df = pd.merge(df, df_returns, on=['date', 'company_name'])


# In[29]:


# to check the column required for the dataset are on the merged dataset 
merged_df.info()


# In[30]:


# changing from type object to float 
merged_df['RET'] = merged_df['RET'].astype(float)


# In[31]:


# Calculating daily return and create labels for direction of returns
merged_df['return_direction'] = np.where(merged_df['RET'] > 0, 2, 
                                np.where(merged_df['RET'] == 0, 1, 0))


# In[32]:


# checking the values for each return direction
merged_df['return_direction'].value_counts()


# ## CALCULATING CORRELATION BETWEEN SENTIMENT AND RETURNS

# ### Calculation of Sentiment vs Returns on the same day (FinBERT)

# In[33]:


# Function to calculate Pearson correlation for each company
def calculate_pearson_correlation_finbert(df, company_name):
    company_df = df[df['company_name'] == company_name]
    correlations = company_df[['sentiment_finbert', 'return_direction']].corr(method='pearson')
    return correlations.loc['sentiment_finbert', 'return_direction']

# Get unique company names
companies = merged_df['company_name'].unique()

# Calculate Pearson correlation for each company
pearson_correlations = {}
for company in companies:
    pearson_correlations[company] = round(calculate_pearson_correlation_finbert(merged_df, company), 4)

# Display the Pearson correlations for each company
for company, correlation in pearson_correlations.items():
    print(f"Pearson correlation for {company}:\n{correlation}\n")


# ### Calculation of Sentiment vs Returns on the same day (VADER)

# In[34]:


# Function to calculate Pearson correlation for each company
def calculate_pearson_correlation_VADER(df, company_name):
    company_df = df[df['company_name'] == company_name]
    correlations = company_df[['sentiment_VADER', 'return_direction']].corr(method='pearson')
    return correlations.loc['sentiment_VADER', 'return_direction']

# Calculate Pearson correlation for each company
pearson_correlations = {}
for company in companies:
    pearson_correlations[company] = round(calculate_pearson_correlation_VADER(merged_df, company), 4)

# Pearson correlations for each company - Display
for company, correlation in pearson_correlations.items():
    print(f"Pearson correlation for {company}:\n{correlation}\n")


# In[35]:


# Function to calculate Pearson correlation for each company
def calculate_pearson_correlation_LM(df, company_name):
    company_df = df[df['company_name'] == company_name]
    correlations = company_df[['sentiment_LM', 'return_direction']].corr(method='pearson')
    return correlations.loc['sentiment_LM', 'return_direction']

# Calculate Pearson correlation for each company
pearson_correlations = {}
for company in companies:
    pearson_correlations[company] = round(calculate_pearson_correlation_LM(merged_df, company), 4)

# Display the Pearson correlations for each company
for company, correlation in pearson_correlations.items():
    print(f"Pearson correlation for {company}:\n{correlation}\n")


# ### Calculation of Today's Sentiment VS Tomorrow's Return

# ### Using FinBERT sentiment labels

# In[36]:


# done to align today's sentiment with tomorrow's returns  
merged_df['returns_shifted'] = merged_df['return_direction'].shift(-1)

def calculate_pearson_correlation_FinBERT_shifted(merged_df, company_name):
    company_df = merged_df[merged_df['company_name'] == company_name]
    
    # Check if 'returns_shifted' is in the company_df
    if 'returns_shifted' not in company_df.columns:
        raise KeyError("The 'returns_shifted' column is not in the dataframe.")
    
    correlation = company_df[['sentiment_finbert', 'returns_shifted']].corr(method='pearson')
    return correlation.loc['sentiment_finbert', 'returns_shifted']

merged_df['returns_shifted'] = merged_df.groupby('company_name')['return_direction'].shift(-1)


# Pearson correlation for each company
pearson_correlations = {}
for company in companies:
    pearson_correlations[company] = round(calculate_pearson_correlation_FinBERT_shifted(merged_df, company), 4)

# Pearson correlation for each company - Display
for company, correlation in pearson_correlations.items():
    print(f"Pearson correlation for {company}:\n{correlation}\n")


# ### Using VADER as sentiment labels 

# In[37]:


def calculate_pearson_correlation_VADER_shifted(merged_df, company_name):
    company_df = merged_df[merged_df['company_name'] == company_name]
        
    if 'returns_shifted' not in company_df.columns:
        raise KeyError("The 'returns_shifted' column is not in the dataframe.")
    
    correlation = company_df[['sentiment_VADER', 'returns_shifted']].corr(method='pearson')
    return correlation.loc['sentiment_VADER', 'returns_shifted']

merged_df['returns_shifted'] = merged_df.groupby('company_name')['return_direction'].shift(-1)

# Pearson correlation for each company
pearson_correlations = {}
for company in companies:
    pearson_correlations[company] = round(calculate_pearson_correlation_VADER_shifted(merged_df, company), 4)

# Pearson correlations for each company - Display
for company, correlation in pearson_correlations.items():
    print(f"Pearson correlation for {company}:\n{correlation}\n")


# ### Using LM as sentiment label

# In[38]:


def calculate_pearson_correlation_LM_shifted(merged_df, company_name):
    company_df = merged_df[merged_df['company_name'] == company_name]
    
    if 'returns_shifted' not in company_df.columns:
        raise KeyError("The 'returns_shifted' column is not in the dataframe.")
    
    correlation = company_df[['sentiment_LM', 'returns_shifted']].corr(method='pearson')
    return correlation.loc['sentiment_LM', 'returns_shifted']

merged_df['returns_shifted'] = merged_df.groupby('company_name')['return_direction'].shift(-1)

# Pearson correlation for each company
pearson_correlations = {}
for company in companies:
    pearson_correlations[company] = round(calculate_pearson_correlation_LM_shifted(merged_df, company), 4)

# Displaying the correlation for each company
for company, correlation in pearson_correlations.items():
    print(f"Pearson correlation for {company}:\n{correlation}\n")


# In[41]:


## removing 0 or neutral returns as the study focuses on checking positive or negative returns

merged_df = merged_df[merged_df['return_direction'] != 1]


# In[42]:


## checking the value counts of the return direction
merged_df['return_direction'].value_counts()


# ## PREDICTION MODEL - SVM AND LSTM 

# ## SVM - SUPERVISED MACHINE LEARNING CLASSFICATION APPROACH

# ## Using FinBERT sentiments as feature variables (3 Classes : positive, neutral, negative)

# ### Contains model and evaluation metrics - Performance Metrics and Confusion Matrix

# In[44]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Setting random seeds for full reproducibility for the study
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# feature variables (X) and Target Variable (y)
X = merged_df[['positive', 'neutral', 'negative']]
y = merged_df['return_direction']

# Replacing 2 with 1 in the 'return_direction' column to ensure the ROC-AUC score can be done 
y = y.replace(2, 1)

# Train-test split with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardising the features with consistent scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM classifier - Training
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Test Dataset - Prediction
y_pred = svm_model.predict(X_test_scaled)

# Metrics Calculation - Accuracy, Performance Metrics, Confusion Matrix
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# ### ROC-AUC CURVE

# In[45]:


from sklearn.metrics import roc_curve, roc_auc_score, auc

# Calculate the ROC-AUC curve
y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities of the positive class

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()


# ## Using FinBERT sentiments as feature variables (positive, neutral, negative) (SMOTE)

# In[46]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Setting random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Feature variables (X) and Target Variable (y)
X = merged_df[['positive', 'neutral', 'negative']]
y = merged_df['return_direction']

# Replacing 2 with 1 in the 'return_direction' column to make the ROC-AUC score
y = y.replace(2, 1)

# Train-test split with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE application to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Training the classifier
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

# Prediction on Test Dataset 
y_pred = svm_model.predict(X_test_scaled)

# Metrics calculation using Accuracy, performance metrics and confusion matrix 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# ### ROC-AUC CURVE 

# In[47]:


from sklearn.metrics import roc_curve, roc_auc_score, auc

# Calculate the ROC-AUC curve
y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities of the positive class

# ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()


# ## Using VADER Sentiment Labels as feature variables

# In[50]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set random seeds for full reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Prepare the feature matrix (X) and target vector (y)
X = merged_df['sentiment_VADER'].values.reshape(-1, 1)
y = merged_df['return_direction']

# Replacing 2 with 1 in the 'return_direction' column
y = y.replace(2, 1)

# Train-test split with a fixed random state for reproducibility
X_train_VADER, X_test_VADER, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier with a fixed random state for reproducibility
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_VADER, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_VADER)

# Metrics calculation - Confusion Matrix
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# ### ROC-AUC CURVE 

# In[51]:


from sklearn.metrics import roc_curve, roc_auc_score, auc

# Calculate the ROC-AUC curve
y_proba = svm_model.predict_proba(X_test_VADER)[:, 1]  # Probabilities of the positive class

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()


# ## Using VADER Sentiment Labels as feature variables (With SMOTE)

# In[52]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seeds for full reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_VADER, y_train)

# Train the SVM classifier without scaling
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = svm_model.predict(X_test_VADER)

# Metrics calculation - Accuracy, Performance Metrics, Confusion Matrix
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# ###  ROC-AUC CURVE

# In[53]:


from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Calculate the ROC-AUC curve
y_proba = svm_model.predict_proba(X_test_VADER)[:, 1]  # Probabilities of the positive class

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()


# ## LSTM  - Neural Network Approach 

# ## Using FinBERT sentiments as feature variables (3 Classes : positive, neutral, negative)

# In[57]:


import os
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Setting the random seeds to ensure reproducibility as per the study 
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configuring the TensorFlow session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Feature Variables - sentiment variables (FinBERT) 
X = merged_df[['positive', 'neutral', 'negative']].values

# target variable - stock price return 
y = merged_df['return_direction'].values

# Converting the labels to binary (0 and 1)
y = (y == 2).astype(int) 

# timesteps of 5 days (rolling window approach)
timesteps = 5
X_reshaped = []
y_reshaped = []

for i in range(len(X) - timesteps + 1):
    X_reshaped.append(X[i:i + timesteps])
    y_reshaped.append(y[i + timesteps - 1])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# Data Split - 80% training and 20% test data 
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Building the LSTM model of 3 layers with Batch Normalisation and Dropout Layers 
model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, X.shape[1]), activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # end output is a Binary classification output

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training 
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Model Evaluation 
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# metrics calculating using Confusion Matrix and Performance Metrics (Accuracy, Precision, recall, f1 score)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred_classes))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_classes))


# ## Using VADER Sentiment Labels as feature variables

# In[61]:


import os
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Setting the random seeds to ensure reproducibility as per the study 
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configuring the TensorFlow session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Feature Variables - sentiment variables (VADER)
# target variable - stock price return 
X = merged_df[['sentiment_VADER']].values 
y = merged_df['return_direction'].values

# Convert labels to binary (0 and 1)
y = (y == 2).astype(int)  # Convert 2 (positive) to 1, and 0 (negative) remains 0

# timesteps of 5 days (rolling window approach)
timesteps = 5
X_reshaped = []
y_reshaped = []

for i in range(len(X) - timesteps + 1):
    X_reshaped.append(X[i:i + timesteps])
    y_reshaped.append(y[i + timesteps - 1])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Reshape the data to 2D for SMOTE
X_train_2D = X_train.reshape(X_train.shape[0], -1)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_2D, y_train)

# Reshaping to original 3D shape after SMOTE
X_train_resampled = X_train_resampled.reshape(-1, timesteps, 1)

# LSTM Model
model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, 1), activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification output

# Model Compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training
history = model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Model Evaluation 
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# metrics calculating using Confusion Matrix and Performance Metrics (Accuracy, Precision, recall, f1 score)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred_classes))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_classes))


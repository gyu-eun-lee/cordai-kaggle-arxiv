import nltk
import pandas as pd
import random
import re
import swifter

from nltk import word_tokenize, WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nltk.download('punkt')
nltk.download('wordnet')

def deduplicate(df):
    """
    Input:
        : df : unprocessed dataset.
    Output:
        : df : deduplicated dataset.
    """
    # Remove title-summary-term duplicates
    df = df[~df.duplicated(['titles', 'summaries', 'terms'])]
    # Drop rows with null title, non-null duplicated summary, duplicated terms
    drop_rows = df.loc[
        (df['titles'].isna()) 
        & (df['summaries'].notna()) 
        & (df.duplicated(['summaries', 'terms'],keep=False))
    ]
    df.drop(index=drop_rows.index, inplace=True)
    # Drop rows with non-null duplicated title, null summary, duplicated terms
    drop_rows = df.loc[
        (df['titles'].notna()) 
        & (df['summaries'].isna()) 
        & (df.duplicated(['titles', 'terms'],keep=False))
    ]
    df.drop(index=drop_rows.index, inplace=True)

    # Remaining rows with non-null duplicated summaries
    summary_duplicates = df.loc[(df['summaries'].notna()) & (df.duplicated(['summaries'], keep=False))][['titles','summaries','terms']]
    df.drop(index=summary_duplicates.index, inplace=True)

    # Remaining rows with non-null duplicated titles
    title_duplicates = df.loc[(df['titles'].notna()) & (df.duplicated(['titles'], keep=False))][['titles','summaries','terms']]
    df.drop(index=title_duplicates.index, inplace=True)

    return df
    
def impute(df):
    """
    Input:
        : df : deduplicated dataset.
    Output:
        : df : imputed and concatenated dataset.
               Null string entries imputed with ''.
    """
    df = df.fillna('')
    
    return df

def concat(df):
    """
    Input:
        : df : deduplicated and imputed dataset.
    Output:
        : df : titles and summaries columns concatenated to text
    """
    df['text'] = df['titles'] + '.\n' + df['summaries']
    df.drop(['titles','summaries','ids'],axis=1,inplace=True)
    
    return df

def multiHotEncode(df):
    """
    Input:
        : df : deduplicated and imputed dataset.
    Output:
        : df : dataset with multi-hot encoded classes.
    """
    # Terms processing: entries in terms are of type str
    df['terms'] = df['terms'].str.replace('[',"")
    df['terms'] = df['terms'].str.replace(']',"")
    df['terms'] = df['terms'].str.replace("'","")
    df['terms'] = df['terms'].str.replace(" ","")
    # Converting entries from str to list
    df['terms'] = df['terms'].str.strip('()').str.split(',')
    df['terms'] = pd.Series(df['terms'])
    df['terms']

    mlb = MultiLabelBinarizer()
    target_subjects = pd.DataFrame(mlb.fit_transform(df['terms']),
                       columns=mlb.classes_,
                       index=df.index)

    # Replacing "terms" column with multi-hot encoding
    df.drop(['terms'],axis=1,inplace=True)
    df = df.join(target_subjects).reset_index().drop(['index'],axis=1)
    
    return df

def remove_stopwords(text: str):
    words = [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    new_text = " ".join(words)
    return new_text

def regexp_preprocess(text):
    text = re.sub('-', ' ', text) # expand hyphens
    text = re.sub('<.*?>', '', text)   # remove HTML tags
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub(r'\d+\b','', text) # remove numbers
    text = re.sub(r'\b\S{1}\b', '', text) # remove isolated non-whitespace characters
    text = re.sub('^(https:|http:|www\.)\S*', '', text) # remove URLs
    text = text.lower() # lower case, .upper() for upper 
    return text

def lemmatize(text, lemmatizer):
    word_list = word_tokenize(text)
    word_list =  [lemmatizer.lemmatize(word, 'n') for word in word_list]
    word_list =  [lemmatizer.lemmatize(word, 'v') for word in word_list]
    word_list =  [lemmatizer.lemmatize(word, 'a') for word in word_list]
    word_list =  [lemmatizer.lemmatize(word, 'r') for word in word_list]
    word_list =  [lemmatizer.lemmatize(word, 's') for word in word_list]
    return word_list

def process(df, mode):
    lemmatizer = WordNetLemmatizer()
    if mode == 'train':
        df = deduplicate(df)
        df = impute(df)
        df = multiHotEncode(df)
    df = concat(df)
    df['text'] = df['text'].swifter.apply(lambda x: regexp_preprocess(x))
    df['text'] = df['text'].swifter.apply(lambda x: remove_stopwords(x))
    df['text'] = df['text'].swifter.apply(lambda x: lemmatize(x, lemmatizer))
    df.reindex()
    return df
    
# calculate class weights for binary cross entropy loss
def get_pos_weight(y, eps=1e-12):
    num_samples = y.shape[0]
    positive = y.sum().to_numpy()
    negative = num_samples - positive
    pos_weight = (negative / (positive + eps)).tolist()
    return pos_weight

def sample_tokens(tokens, max_length):
    try:
        sample = random.sample(tokens, max_length)
    except ValueError:
        sample = tokens
    return sample

def join_tokens(tokens):
    return ' '.join(tokens)

def random_sampling_augment(df, classes, num_samples):
    augmented_df = pd.DataFrame(columns = df.columns)
    for class_name in classes:
        class_sample = df[df[class_name] == 1].sample(num_samples, replace=True)
        class_sample['text'] = class_sample['text'].apply(lambda x: join_tokens(sample_tokens(x, 32)))
        augmented_df = pd.concat([augmented_df, class_sample], axis=0)
    augmented_df = augmented_df.reset_index()
    return augmented_df
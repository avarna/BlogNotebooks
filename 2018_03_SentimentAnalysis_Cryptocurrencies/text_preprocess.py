
'''
Custom functions
'''

def vectorize_sequences(sequences, dimension):
    '''
    Function to check whether the top_words occur in a given sequence of words (sequences/sentence)
    It just checks whether a word (e.g. 'don't') occurs in a sentence.
    Does matter where in a sentence it occurs and how many times it occurs.

    Can only be used on small data. Keras gives Error 9 (killed) due to memory issues for large data (<300,000 samples)
    Use it ONLY with Dense neural network. Does NOT WORK with 1D CNN
    '''
    import numpy as np
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # i=sentence number (row number) in a matrix, sequence=word number in a row/sentence
        results[i, sequence] = 1.

    return results



def get_glove(glove_dir, word_index, embed_dim=100, max_words=10000):
    '''
    Function to get embedding matrix, based on glove.
    This matrix can be used to initial weights of the Embedding layer of CNN
    Refer textbook 'Deep learning with Python' by F. Chollet (Pg 172)
    '''
    import os
    import numpy as np
    embeddings_index = {}
    f = open(glove_dir)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()
    print('Found ',len(embeddings_index),' words in glove')

    embed_mat = np.zeros((max_words, embed_dim))
    
    count=0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i<max_words:
            if embedding_vector is not None: # Words not found in embedding index will be all-zeros.
                count = count+1
                embed_mat[i-1] = embedding_vector
    
    print(count,' words of the ',min(max_words,len(word_index)),' input words are found in Glove.')

    return embed_mat



def strip_hash(sent):
    '''
    Function to remove hashtag words appearing at the end of a sentence
    Hashtag at the end of the sentences are usually useless
    '''
    new_sent = sent.split()
    for i in range(len(sent.split())-1, -1, -1):
        if (sent.split()[i].startswith('#') or sent.split()[i].startswith('$')):
            del new_sent[i]
        else:
            return ' '.join(new_sent)
            break


def clean_data(data):
    import re
    # remove all the links (pic,http,https) first. Otherwise it creates problems
    data['Text'] = data['Text'].apply(lambda x: ' '.join(re.sub("pic.(\S+)"," ", x).split())) # Removes picture links
    data['Text'] = data['Text'].apply(lambda x: ' '.join(re.sub("http.(\S+)"," ", x).split())) # Remove links with http
    data['Text'] = data['Text'].apply(lambda x: ' '.join(re.sub("https.(\S+)"," ", x).split()))

    data['Text'] = data['Text'].apply(lambda x: ' '.join(re.sub("@(\w+)"," ", x).split())) # Remove words with usernames
    data['Text'] = data['Text'].apply(lambda x: x.replace('…','')) # Remove unwanted, frequently occuring symbols
    #data['Text'] = data['Text'].apply(lambda x: x.replace('.','')) # Don't remove '.' as it can carry important info
    #data['Text'] = data['Text'].apply(lambda x: x.replace('-','')) # Don't remove '-' as it might carry important info
    data['Text'] = data['Text'].apply(lambda x: x.replace('via',''))
    data['Text'] = data['Text'].apply(lambda x: strip_hash(x))
    # The above commands creates empty entries in 'Text' in some cases. Drops the rows with empty text
    data.dropna(inplace=True)

    return data


def remove_symbol(data):
    # Function to remove punctuation
    # data is a STRING
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    data_list = tokenizer.tokenize(data) # remove punctuations
    data_out = ' '.join(data_list) # Convert list back to string
    return data_out


def remove_stopword(data):
    # Function to remove stop words
    from nltk.corpus import stopwords
    data = data.lower().split() # Make words case insensitive and Tokeniz
    clean = [word for word in data if word not in stopwords.words('english')]
    data_out = ' '.join(clean) # Convert list back to string
    return data_out








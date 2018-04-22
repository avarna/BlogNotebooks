
'''
Custom metrics functions
'''

'''
CAUTION 1): Accuracy measure might be misleading for imbalanced datasets. Precision & recall are better
CAUTION 2): recall and precision were removed from keras since it's not accurate

reference for the codes of recall & precision: 
https://github.com/fchollet/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7
'''

def recall(y_true, y_pred):
    import keras.backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    import keras.backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precs = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    f1 = (2*precs*recal)/(precs+recal)
    return f1






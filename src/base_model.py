
import tensorflow as tf
from transformers import TFBertModel,BertTokenizer
import pandas as pd
import numpy as np
from numpy import array
import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report


class BASE_SENTIMENT_MODEL():
    def __init__(self,train_path="data\\train_sent_emo.csv",
            test_path="data\\test_sent_emo.csv",ver_path="data\\dev_sent_emo.csv",
            result_path='results\\model_results_context.csv',
            save_path='models\\model_context\\Weights_context',
            length_tokenization=32):
        
        self.train_path=train_path
        self.test_path=test_path
        self.ver_path=ver_path
        self.result_path=result_path
        self.save_path=save_path
        self.length=length_tokenization

        self.columns=['Utterance','label','Speaker']
        self.encoded_dict={'negative' :0, 'neutral' :1, 'positive': 2}

        self.tz = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        
        self.model=self.create_model()
    
    #use this function to train the model 
    def train_model(self) :
        test_data,y_test=self.get_data(self.ver_path)
        train_data,y_train=self.get_data(self.train_path)

        full_data=self.regroup_data(train_data,test_data)
        target_data=np.concatenate((y_train, y_test))

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=5, shuffle=True)

        # K-fold Cross Validation model evaluation
        fold_no = 1
        print("begin training")

        results=[]
        time_elapsed=[]

        start = time.time()
        for train, test in kfold.split(full_data['review']):
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
        
            x_train=self.tokenize(full_data['review'].iloc[train].tolist())
            x_test=self.tokenize(full_data['review'].iloc[test].tolist())
            y_train=target_data[train]
            y_test=target_data[test]
        
            train_history=self.fit_model(x_train,x_test,y_train,y_test)
            # Increase fold number
            fold_no = fold_no + 1
            print("predicting..")
            metrics=self.predict_test_data()
            results.append(metrics)
            end=time.time()
            elapsed=(end-start)
            time_elapsed.append(elapsed)
        
        # save
        self.save_results(results)
        self.model.save_weights(self.save_path)


    #used to predict single sentences, will be used when the model is ready
    def predict_sentiment(self,sentence):
        
        tokenized_sentence=self.other_tokenize(sentence)
        predicted_raw = self.model.predict({'input_ids':tokenized_sentence['input_ids'],
                'attention_mask':tokenized_sentence['attention_mask']})

        y_predicted = np.argmax(predicted_raw, axis = 1)
        print(y_predicted)
        return y_predicted


    '''the rest of functions are used by the functions above'''
    #csv to pandas
    def get_data(self,path) :
        data = pd.read_csv(path)
        new_column, y=self.sentiment_to_int(data['Sentiment'])
        data['label']=new_column
        data=data[self.columns]
        data=data.rename(columns={"Utterance":"review"})
        return data,y

    #string sentiments to one-hot-encoding
    def sentiment_to_int(self,column) :
        new_column=[]
        for element in column :
            if element=='positive' :
                new_column.append(2)
            elif element=='neutral' :
                new_column.append(1)
            else :
                new_column.append(0)
        new_column=array(new_column)
        encoded=to_categorical(new_column)
        encoded_col=encoded.tolist()
        #inverted = argmax(encoded[0])
        return encoded_col,encoded

    #regroup validation and training data, used for k-crossfold validation
    def regroup_data(self,df1,df2) :
        frames = [df1, df2]

        result = pd.concat(frames)
        return result


    # Tokenize the input (takes some time) 
    # here tokenizer using from bert-base-uncased
    def tokenize(self,data) :
        x_test=self.tz(
        text=data,
        add_special_tokens=True,
        max_length=self.length,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)


        return x_test


    #used to test singular strings, not whole dataset
    def other_tokenize(self, data) :
        encoded = self.tz.encode_plus(
        text=data,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = self.length,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        return_tensors = 'tf',  # ask the function to return tf tensors
        )
        return encoded



    def create_model(self) :
        input_ids = Input(shape=(self.length,), dtype=tf.int32, name="input_ids")
        input_mask = Input(shape=(self.length,), dtype=tf.int32, name="attention_mask")
        embeddings = self.bert(input_ids,attention_mask = input_mask)[0] 
        out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
        out = Dense(128, activation='relu')(out)
        out = tf.keras.layers.Dropout(0.1)(out)
        out = Dense(32,activation = 'relu')(out)
        y = Dense(3,activation = 'softmax')(out)
        model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
        model.layers[2].trainable = True
        model=self.compile_model(model)
        return model


    def compile_model(self,model,lr=5e-05,eps=1e-08) :
        optimizer = Adam(
        learning_rate=lr, # this learning rate is for bert model , taken from huggingface website 
        epsilon=eps,
        decay=0.01,
        clipnorm=1.0)
        # Set loss and metrics
        loss =CategoricalCrossentropy(from_logits = True)
        metric = CategoricalAccuracy('balanced_accuracy'),
        # Compile the model
        model.compile(
            optimizer = optimizer,
            loss = loss, 
            metrics = metric)
        return model


    def fit_model(self,x_train,x_test,y_train,y_test) :
        train_history = self.model.fit(
        x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
        y = y_train,
        validation_data = (
        {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
        ),
        epochs=1,
        batch_size=36
        )
        return train_history


    #load an already saved model
    def load_model(self) :
        self.model.load_weights(self.save_path)


    #decode the one-hot-encoding
    def original_sentiment(self,y_test) :
        y_true=[]
        for element in y_test :
            if str(element)=='[1. 0. 0.]' :
                y_true.append(0)
            elif str(element)=='[0. 1. 0.]' :
                y_true.append(1)
            else :
                y_true.append(2)
        
        return y_true


    #used to test our model on the test_data, then save the result
    def predict_test_data(self) :
        test_data,y_test=self.get_data(self.test_path)
        x_test=self.tokenize(test_data['review'].to_list())
        
        predicted_raw = self.model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})

        y_predicted = np.argmax(predicted_raw, axis = 1)
        y_true = self.original_sentiment(y_test)

        results=classification_report(y_true, y_predicted,target_names=self.encoded_dict.keys(),output_dict=True)
        return results


    def save_results(self,results) :
        df = pd.DataFrame(results).transpose()
        df.to_csv(self.result_path)
        print(df)


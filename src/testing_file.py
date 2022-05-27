from base_model import BASE_SENTIMENT_MODEL
  


if __name__ == '__main__':

    sentiment_model=BASE_SENTIMENT_MODEL()
    #sentiment_model.train_model()
    sentiment_model.load_model()
    sentiment_model.predict_sentiment("the waiter is bad, im not coming back again")

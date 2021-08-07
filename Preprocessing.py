def data_preprocessing(review):
    
  # data cleaning
   #removing of html tags
  review = re.sub(re.compile('<.*?>'), '', review)
   #taking only words
  review =  re.sub('[^A-Za-z0-9]+', ' ', review)
  
  # lowercase
   # converting every uppercase words into lower case
  review = review.lower() 
  
  # tokenization
   # converts review to tokens, splitting the review into small words.
  tokens = nltk.word_tokenize(review) 
  
  # stop_words removal
   #removing stop words which are generally not useful.
  review = [word for word in tokens if word not in stop_words] 
  
  # lemmatization
   # grouping of the similar inflected form of words, which therefore can be analyzed as a single item.
  review = [lemmatizer.lemmatize(word) for word in review]
  
  # join words in preprocessed review
  review = ' '.join(review)
  
  return review

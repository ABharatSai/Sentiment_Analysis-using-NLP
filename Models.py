# After preprocssing we convert the preprocessed text into vectors which gives the words their numerical form. Which will help the machine to understand.
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer(min_df=20)

X_train_vec = vector.fit_transform(X_train['preprocessed_review'])
X_test_vec = vector.transform(X_test['preprocessed_review'])

print(X_train_vec.shape, X_test_vec.shape)

# It is also similar to above but it does give good results than above.
from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(min_df=20)

X_train_vec = vector.fit_transform(X_train['preprocessed_review'])
X_test_vec = vector.transform(X_test['preprocessed_review'])

print(X_train_vec.shape, X_test_vec.shape)

# HERE I am using the Naive Bayees Algorithm from sklearn.
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print(accuracy_score(y_test, y_pred))

# HERE I am using the Random Forest Algorithm from sklearn.
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print('Test Accuracy: ', accuracy_score(y_test, y_pred))

# HERE I am using Logistic Regression Algorithm from sklearn.
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print('Test Accuracy: ', accuracy_score(y_test, y_pred))

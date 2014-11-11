from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import svm

y = [11, 11]
d =   [{'category': 'Other', 'city': 'Atlanta', 'started_at': '3/21/2011', 'discount_pct': 50, 'value': 30, 'sold_out': 0, 'weekday_start': 'Mon', 'duration': 7, 'you_save': 15, 'family_edition': 1}, {'category': 'Entertainment', 'city': 'New York City', 'started_at': '3/22/2011', 'discount_pct': 50, 'value': 40, 'sold_out': 0, 'weekday_start': 'Tue', 'duration': 1, 'you_save': 20, 'family_edition': 0}]


vec = feature_extraction.DictVectorizer()

vec_train_data_topic = vec.fit_transform(d)

#clf = svm.SVR()
clf = linear_model.LinearRegression()
clf.fit(vec_train_data_topic,y)

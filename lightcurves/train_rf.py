import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Training_Features_2class.csv')

del df['Macho_id']
y = df['label']
del df['label']
X = df

clf = RandomForestClassifier(n_estimators=20, max_features=5)
clf = clf.fit(X,y)

scores = cross_val_score(clf, X, y)

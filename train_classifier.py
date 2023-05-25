import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])
print(data)
model = RandomForestClassifier()

model.fit(data, labels)



f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
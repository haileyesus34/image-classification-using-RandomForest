from img2vec_pytorch import Img2Vec
from PIL import Image
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

#prepare the data 

dataPath = './data'
trainPath = './data/train'
valPath = './data/val'


img_2_vec = Img2Vec()
data = {}

for j, dir_ in enumerate([trainPath, valPath]):
    features = []
    labels = []
    print()
    for category in os.listdir(dir_): 
        print(category)
        for imagePath in os.listdir(os.path.join(dir_, category)): 
            img_path = os.path.join(dir_, category, imagePath)
            img = Image.open(img_path)
            img = img.convert('RGB')

            feature = img_2_vec.get_vec(img)

            features.append(feature)
            labels.append(category)

    data[['train_data', 'val_data'][j]] = features
    data[['train_labels', 'val_labels'][j]] = labels


# load model 
model =  RandomForestClassifier()

# train model 
model.fit(data['train_data'], data['train_labels'])

y_pred = model.predict(data['val_data'])

acc = accuracy_score(y_pred, data['val_labels'])

with open('./model.p', 'wb') as f: 
    pickle.dump(model, f)
    f.close()



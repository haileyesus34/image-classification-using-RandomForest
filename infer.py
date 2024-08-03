from img2vec_pytorch import Img2Vec 
from PIL import Image 
import pickle 


img_2_vec = Img2Vec()

with open('./model', 'rb') as f: 
    model = pickle.load(f)

img_path = '/Users/hanna m/machinelearning/deep_learning/cv/classification/data/train/cloudy/cloudy2.jpg'

img = Image.open(img_path)

features = img_2_vec.get_vec(img)

pred = model.predict([features])

print(pred)

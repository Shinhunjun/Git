import os
import cv2
import numpy as np

main = '/Users/hunjunsin/Desktop/python/Kaggle/genderclassification/gender_dataset'

men = os.path.join(main, 'men')
women = os.path.join(main, 'women')

def preprocess(directory, label):
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"unable to read image {img_path}")
                    continue
                
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
                 
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")
    
    return images, labels


mimages, mlabels = preprocess(men, 0)
woimages, wolabels = preprocess(women, 1)

from tensorflow.keras.utils import to_categorical

images = np.array(mimages + woimages)
labels = np.array(mlabels + wolabels)

images = images / 255.0
labels = to_categorical(labels, num_classes=2)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 42)


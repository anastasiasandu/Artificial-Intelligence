import tensorflow as tf
import numpy as np
import os
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def read(img):
    try:
        im = Image.open(img)                                                             #fisierul din care citesc imaginea
        im_resized = im.resize((150, 150))                                               #deschid fisierul
        return np.array(im_resized)                                                      #transform in np.array
    finally:
        im.close()                                                                       #inchid fisierul

train_img = []                                                                           #imaginile de training
labels = []                                                                              #etichetele imaginilor de training
cnt = 0
with open('/kaggle/input/unibuc-brain-ad/data/labels.txt', 'r') as f:
    next(f)
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'                                                            #construiesc id-ul imaginii
        img = read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data', id))          #caut imaginea cu id-ul respectiv
        train_img.append(img)
        labels.append(int(label))
        cnt += 1
        if cnt == 15000:                                                                 #primele 15000 de poze sunt de training
            break

train_img = np.array(train_img)                                                          #il convertesc intr-un numpy array
train_img = train_img.astype('float32')                                                  #il convertesc la tipul de date float32
labels = np.array(labels)                                                                #il convertesc intr-un numpy array

v_img = []                                                                               #analog pentru datele de validare
v_label = []
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as f:
    next(f)
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'
        img = read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data', id))
        v_img.append(img)
        v_label.append(int(label))
        cnt += 1
        if cnt == 17000:                                                                 #datele de validare sunt de la 15000 la 17000
            break

v_img = np.array(v_img)
v_img = v_img.astype('float32')
v_label = np.array(v_label)

with open('/kaggle/input/unibuc-brain-ad/data/sample_submission.txt', 'r') as f:         #analog si pentru datele de test
    next(f)
    v_imgfinal=[]
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'
        img=read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data',id))
        v_imgfinal.append(img)
v_imgfinal = np.array(v_imgfinal)
v_imgfinal = v_imgfinal.astype('float32')

train_img /= 255.0                                                                      #normalizez datele
v_img /= 255.0
v_imgfinal /= 255.0


train_data = ImageDataGenerator(                                                        #augmentez datele
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

train_generator = train_data.flow(                                                      #aplic transformarile pe datele de training
    train_img, labels,
    batch_size=32)

v_data = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False )

val_generator = v_data.flow(                                                            #aplic transformarile pe datele de validare
    v_img, v_label,
    batch_size=32)

model = tf.keras.models.Sequential()                                                    #construiesc reteaua neurala
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])     #compilez modelul


history = model.fit(train_generator,                                                  #antrenez modelul
                    steps_per_epoch=len(train_img) // 32,
                    epochs=20,
                    validation_data=val_generator,
                    validation_steps=len(v_img) // 32)

from sklearn.metrics import f1_score

l = model.predict(v_img)                                                              #fac predictia pe datele de validare
l = (l > 0.34).astype(int)                                                            #transform in etichete binare
#print(f1_score(l, v_label))


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

acuratete = accuracy_score(v_label, l)                                                #acuratetea modelului
raport = classification_report(v_label, l)                                            #raportul de clasificare
matrice = confusion_matrix(v_label, l)                                                #matricea de confuzie

print("Classification accuracy:", acuratete)
print("Classification report:\n", raport)
print("Confusion matrix:\n", matrice)

fig, ax = plt.subplots(figsize=(6, 4))                                                #creez o diagrama goala de dimensiunea 6*4 si axele acesteia
sns.heatmap(matrice, annot=True, cmap="PuRd", fmt="d", ax=ax)                         #construiesc diagrama matricii
ax.set_xlabel('Predicted label')                                                      #setez eticheta pentru axa x a diagramei
ax.set_ylabel('True label')                                                           #setez eticheta pentru axa y a diagramei
plt.show()

l = model.predict(v_imgfinal)                                                         #fac predictia pe datele de test
l = (l > 0.34).astype(int)                                                            #transform in etichete binare
#print(sum(l))


with open('naive_bayes_predictions18.csv', 'w') as f:                                 #scriu predictia in fisierul .csv
    f.write('id,class\n')
    c = 17001
    for k in l:
        f.write(f'0{c},')                                                             #formatul cerut
        f.write(str(k[0]))
        f.write('\n')
        c = c + 1


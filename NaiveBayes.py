import numpy as np
from PIL import Image
import os


def read(img):  # fisierul din care citesc imaginea
    try:
        im = Image.open(img)  # deschid fisierul
        return np.array(im)  # transform in np.array
    finally:
        im.close()  # inchid fisierul


train_img = []  # imaginile de training
cnt = 0
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as f:
    next(f)
    labels = []  # etichetele imaginilor de training
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'  # construiesc id-ul imaginii
        img = read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data', id))  # caut imaginea cu id-ul respectiv
        train_img.append(img)
        labels.append(label)
        cnt = cnt + 1
        if cnt == 15000:  # primele 15000 de poze sunt de training
            break

labels = np.array(labels)  # il convertesc intr-un numpy array
labels = labels.astype(int)  # il convertesc la tipul de date int
train_img = np.array(train_img)  # il convertesc intr-un numpy array
train_img = train_img.reshape(train_img.shape[0], -1)  # il fac 2-dimensional

v_img = []  # analog pentru datele de validare
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as f:
    next(f)
    v_label = []
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'
        img = read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data', id))
        v_img.append(img)
        v_label.append(label)
        cnt = cnt + 1
        if cnt == 17000:  # datele de validare sunt de la 15000 la 17000
            break

v_img = np.array(v_img)
v_img = v_img.reshape(v_img.shape[0], -1)
v_label = np.array(v_label)
v_label = v_label.astype(int)

with open('/kaggle/input/unibuc-brain-ad/data/sample_submission.txt', 'r') as f:
    next(f)
    v_imgfinal = []  # analog si pentru datele de test
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'
        img = read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data', id))
        v_imgfinal.append(img)

v_imgfinal = np.array(v_imgfinal)
v_imgfinal = v_imgfinal.reshape(v_imgfinal.shape[0], -1)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()  # construiesc modelul
model.fit(train_img, labels)  # il antrenez

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

l = model.predict(v_img)  # fac predictia pe datele de validare

# print(f1_score(l, v_label))

acuratete = accuracy_score(v_label, l)  # acuratetea modelului
raport = classification_report(v_label, l)  # raportul de clasificare
matrice = confusion_matrix(v_label, l)  # matricea de confuzie

print("Classification accuracy:", acuratete)
print("Classification report:\n", raport)
print("Confusion matrix:\n", matrice)

fig, ax = plt.subplots(figsize=(6, 4))  # creez o diagrama goala de dimensiunea 6*4 si axele acesteia
sns.heatmap(matrice, annot=True, cmap="PuRd", fmt="d", ax=ax)  # construiesc diagrama matricii
ax.set_xlabel('Predicted label')  # setez eticheta pentru axa x a diagramei
ax.set_ylabel('True label')  # setez eticheta pentru axa y a diagramei
plt.show()

l = model.predict(v_imgfinal)  # fac predictia pe datele de test

with open('/kaggle/working/naive_bayes_predictions.csv', 'w') as f:  # scriu predictia in fisierul .csv
    f.write('id,class\n')
    c = 17001
    for k in l:
        f.write(f'0{c},')  # formatul cerut
        f.write(str(k))
        f.write('\n')
        c = c + 1


import numpy as np
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def read(img):                                                                             #fisierul din care citesc imaginea
    try:
        im=Image.open(img)                                                                 #deschid fisierul
        return np.array(im)                                                                #transform in np.array
    finally :
        im.close()                                                                         #inchid fisierul

train_img=[]                                                                               #imaginile de training
cnt = 0
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as f:
    next(f)
    labels=[]                                                                              #etichetele imaginilor de training
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'                                                              #construiesc id-ul imaginii
        img=read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data',id))               #caut imaginea cu id-ul respectiv
        train_img.append(img)
        labels.append(label)
        cnt=cnt+1
        if cnt==15000:                                                                     #primele 15000 de poze sunt de training
            break
train_img=np.array(train_img)                                                              #il convertesc intr-un numpy array
labels=np.array(labels)                                                                    #il convertesc intr-un numpy array
labels=labels.astype(int)                                                                  #il convertesc la tipul de date int

v_img=[]                                                                                   #analog si datele de validare
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as f:
    next(f)
    v_label=[]
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'
        img=read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data',id))
        v_img.append(img)
        v_label.append(label)
        cnt=cnt+1
        if cnt==17000:                                                                     #datele de validare sunt de la 15000 la 17000
            break

v_img=np.array(v_img)
v_label=np.array(v_label)
v_label=v_label.astype(int)




with open('/kaggle/input/unibuc-brain-ad/data/sample_submission.txt', 'r') as f:
    next(f)
    v_imgfinal=[]                                                                          #analog si pentru datele de test
    for i in f:
        id, label = i.strip().split(',')
        id = str(id) + '.png'
        img=read(os.path.join('/kaggle/input/unibuc-brain-ad/data/data',id))
        v_imgfinal.append(img)


from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

dict_param = {'n_estimators': randint(50, 200),                                           #construiesc un dictionar cu intervalele pentru hyperparametri
              'max_depth': randint(1, 20),
              'min_samples_split': [2, 5],
              'min_samples_leaf': [1, 2],
              'max_features': ['sqrt', 'log2']}
rf = RandomForestClassifier(random_state=42)
random = RandomizedSearchCV(rf,
                                 param_distributions=dict_param,
                                 n_iter=5,                                               #numarul de combinari aleatorii incercate
                                 cv=5,                                                   #de cate ori e antrenat si evaluat modelul
                                 random_state=42)
random.fit(train_img, labels)                                                            #antrenez modelul
print(random.best_params_)                                                               #returneaza cea mai buna combinatie de hiperparametri

#construiesc modelul
rf = RandomForestClassifier(n_estimators=137, max_depth=19, random_state=42, min_samples_split=2, min_samples_leaf=1, max_features='sqrt')
rf.fit(train_img, labels)                                                                #antrenez modelul

l = rf.predict(v_img)                                                                    #fac predictia pe datele de validare
from sklearn.metrics import f1_score
print(f1_score(preds, v_label))


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

acuratete = accuracy_score(v_label, l)                                                   #acuratetea modelului
raport = classification_report(v_label, l)                                               #raportul de clasificare
matrice = confusion_matrix(v_label, l)                                                   #matricea de confuzie

print("Classification accuracy:", acuratete)
print("Classification report:\n", raport)
print("Confusion matrix:\n", matrice)

fig, ax = plt.subplots(figsize=(6, 4))                                                  #creez o diagrama goala de dimensiunea 6*4 si axele acesteia
sns.heatmap(matrice, annot=True, cmap="PuRd", fmt="d", ax=ax)                           #construiesc diagrama matricii
ax.set_xlabel('Predicted label')                                                        #setez eticheta pentru axa x a diagramei
ax.set_ylabel('True label')                                                             #setez eticheta pentru axa y a diagramei
plt.show()

l = rf.predict(test_img)                                                                #fac predictia pe datele de test

with open('/kaggle/working/naive_bayes_predictions15.csv', 'w') as f:                    #scriu predictia in fisierul .csv
    f.write('id,class\n')
    c=17001
    for k in l:
        f.write(f'0{c},')                                                               #formatul cerut
        f.write(str(k))
        f.write('\n')
        c=c+1
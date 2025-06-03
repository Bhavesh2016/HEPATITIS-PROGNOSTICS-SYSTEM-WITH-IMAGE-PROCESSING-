import tkinter
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pickle
from sklearn.svm import SVC
from PIL import Image, ImageTk 

import cv2
from skimage import color
from skimage.feature import greycomatrix, greycoprops
import scipy.stats as stats
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json


global filename
#global X,Y
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test
global cnn
global labels

labels = ['Hepatitis','Normal_liver']



with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    cnn_classifier = model_from_json(loaded_model_json)
json_file.close()    
cnn_classifier.load_weights("model/model_weights.h5")
cnn_classifier._make_predict_function() 


main = tkinter.Tk()
main.title("Hepatitis") #designing main screen
main.geometry("1300x1200")

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False): #extract glcm features
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = greycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image): #texture features will be extracted using histogram
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    greycoprops(glcm, 'correlation'),
    greycoprops(glcm, 'contrast'),
    greycoprops(glcm, 'energy'),
    greycoprops(glcm, 'homogeneity'),
    greycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index 

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    
    # Check if a folder was selected
    if not filename:
        text.insert(END, "No folder selected. Please try again.\n")
        return
    
    # Validate the dataset folder
    is_valid, message = validate_dataset(filename)
    if not is_valid:
        text.insert(END, "Error: " + message + "\n\n")
        messagebox.showerror("Invalid Dataset", message)
        return
    else:
        text.insert(END, "Dataset validation successful!\n")
        text.insert(END, message + "\n\n")
    
    # Existing logic for showing dataset details
    text.insert(END, filename + ' Loaded\n\n')    
    text.insert(END, "Different Groups Found in Dataset: " + str(labels) + "\n\n") 
    text.insert(END, "Total Groups are: " + str(len(labels)))

# Validation function to be added at the top of your script
def validate_dataset(folder_path):
    """
    Validate the uploaded dataset folder.
    """
    required_labels = ['Hepatitis', 'Normal_liver']  # Expected subfolders
    for label in required_labels:
        label_path = os.path.join(folder_path, label)  # Construct subfolder paths
        if not os.path.exists(label_path) or not os.path.isdir(label_path):
            return False, f"Missing required folder: '{label}'"
        valid_files = [f for f in os.listdir(label_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not valid_files:
            return False, f"No valid image files found in folder: '{label}'"
    return True, "Dataset is valid"
  
def featuresExtraction():
    global filename
    global X,Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    class_label = getID(name)
                    features = extract_features(img)
                    Y.append(class_label)
                    X.append(features)
                    print(name+" "+root+"/"+directory[j]+" "+str(features.shape)+" "+str(class_label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X",X)
        np.save("model/Y",Y)
    X = X.astype('float32')
    X = X/255 #features normalization
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Extracted GLCM & Texture Features : "+str(X[0])+"\n\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Dataset train & test split. 80% dataset images used for training and 20% for testing\n")
    text.insert(END,"80% training images : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% testing images : "+str(X_test.shape[0])+"\n\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show() 

def runSVM():
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    
    if os.path.exists('model/svm.txt'):
        with open('model/svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
        file.close()        
    else:
        # Initialize SVC with probability=True
        svm_cls = SVC(probability=True)
        svm_cls.fit(X_train, y_train)
        with open('model/svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
        file.close()
    
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)





def runKNN():
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)

    if os.path.exists('model/knn.txt'):
        with open('model/knn.txt', 'rb') as file:
            knn_cls = pickle.load(file)
        file.close()
    else:
        knn_cls = KNeighborsClassifier(n_neighbors=3)  # You can change n_neighbors as needed
        knn_cls.fit(X_train, y_train)
        with open('model/knn.txt', 'wb') as file:
            pickle.dump(knn_cls, file)
        file.close()
    
    predict_knn = knn_cls.predict(X_test)
    calculateMetrics("KNN", predict_knn, y_test)



def runCNN():
    global X_train, X_test, y_train, y_test, X, Y, cnn
    global accuracy, precision,recall, fscore
    text.delete('1.0',END)
    Y1 = to_categorical(Y)
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y1, test_size=0.2)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        json_file.close()    
        cnn.load_weights("model/model_weights.h5")
        cnn._make_predict_function()   
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1], XX.shape[2], XX.shape[3]), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim = 256, activation = 'relu'))
        cnn.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn.fit(XX, Y1, batch_size=12, epochs=10, shuffle=True, verbose=2)
        cnn.save_weights('model/model_weights.h5')            
        model_json = cnn.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn.summary())
    predict = cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    calculateMetrics("Propose CNN", predict, y_test)






def predict():
    global cnn
    filename = filedialog.askopenfilename(initialdir="testImages")
    
    # Load and preprocess the new test image
    img = cv2.imread(filename)
    test = []
    img_resized = cv2.resize(img, (64, 64))
    features = extract_features(img_resized)
    test.append(features)
    test = np.asarray(test).astype('float32') / 255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    
    # Predict the class and get confidence scores
    prediction = cnn.predict(test)
    predicted_label = np.argmax(prediction)  # Get the class index
    confidence_score = prediction[0][predicted_label] * 100  # Confidence as a percentage
    
    # Generate the message
    if predicted_label == 0:  # Hepatitis
        message = f"{confidence_score:.2f}% Liver is affected with Hepatitis."
    else:  # Normal Liver
        message = f"{confidence_score:.2f}% Liver is healthy."
    
    # Display the prediction and confidence score on the image
    img_display = cv2.imread(filename)
    img_display = cv2.resize(img_display, (800, 400))
    cv2.putText(img_display, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show the image with the message
    cv2.imshow('Prediction Result', img_display)
    cv2.waitKey(0)



bg_img = Image.open("image.jpg")  
bg_img = bg_img.resize((1300, 1200), Image.ANTIALIAS)  

bg_img_tk = ImageTk.PhotoImage(bg_img)

bg_label = Label(main, image=bg_img_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)



font = ('times', 16, 'bold')
title = Label(main, text='Hepatitis Prognostics System with Image Processing')
title.config(bg='steel blue', fg='ghost white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=150)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload  Dataset", command=uploadDataset)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)  

featuresButton = Button(main, text="Extract Features", command=featuresExtraction)
featuresButton.place(x=50,y=220)
featuresButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=50,y=290)
svmButton.config(font=font1)


knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=50,y=360)
knnButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=50,y=430)
cnnButton.config(font=font1)



predictButton = Button(main, text="Prediction of Hepatitis Disease", command=predict)
predictButton.place(x=50,y=500)
predictButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()







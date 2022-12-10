import tkinter as tk
import tkinter.font as font
import tkinter.ttk as ttk
from tkinter import *
import tensorflow as tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from scipy.spatial import distance as dist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
from pygame import mixer
import numpy as np
import argparse
import imutils
import time
import cv2
import os





print("\n\n \tOpening...\n\n")

#designing the template using tkinter library
window = tk.Tk()  
window.title("project1")
window.geometry('1920x1080')

window.configure(background ='lightcyan')

bg = PhotoImage(file = r'C:\Users\lenovo\anaconda3\mainproject\bgimage.png')
  
#Show image using label
label1 = Label( window, image = bg)
label1.place(x = 0, y = 0)


window.grid_rowconfigure(0, weight = 1) 
window.grid_columnconfigure(0, weight = 1) 
message = tk.Label( 
    window, text ="Face Mask   &   Social Distance Detection",  
    bg ="lightsalmon", fg = "black", width = 50,  
    height = 3, font = ('times', 30, 'bold'))
message.place(x = 75, y = 20)






def face_mask():
    def detect_and_predict_mask(frame, faceNet, maskNet):
            # grab the dimensions of the frame and then construct a blob
            # from it
            (h, w) = frame.shape[0:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                    (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()

            # initialize our list of faces, their corresponding locations,
            # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []

            # loop over the detections
            for i in range(0, detections.shape[1]):
                    # extract the confidence (i.e., probability) associated with
                    # the detection
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the confidence is
                    # greater than the minimum confidence
                    if confidence > 0.5:
                            # compute the (x, y)-coordinates of the bounding box for
                            # the object
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # ensure the bounding boxes fall within the dimensions of
                            # the frame
                            (startX, startY) = (max(0, startX), max(0, startY))
                            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                            # extract the face ROI, convert it from BGR to RGB channel
                            # ordering, resize it to 224x224, and preprocess it
                            face = frame[startY:endY, startX:endX]
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            face = cv2.resize(face, (224, 224))
                            face = img_to_array(face)
                            face = preprocess_input(face)
                            face = np.expand_dims(face, axis=0)

                            # add the face and bounding boxes to their respective
                            # lists
                            faces.append(face)
                            locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                    preds = maskNet.predict(faces)

            # return a 2-tuple of the face locations and their corresponding
            # locations
            return (locs, preds)



    # load face detector from disk
    print(" loading face detector...")
    prototxtPath = 'C:/Users/lenovo/anaconda3/mainproject/face_detector/deploy.prototxt'
    weightsPath = 'C:/Users/lenovo/anaconda3/mainproject/face_detector/res10_300x300_ssd_iter_140000.CAFFEMODEL'
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print(" loading face mask detector model...")
    maskNet = load_model("mask_detector.model")
    #loading alarm
    mixer.init()
    sound = mixer.Sound(r'C:\Users\lenovo\anaconda3\mainproject\alarm.mp3')

    # initialize the video stream 
    print(" starting video stream...")
    vs = VideoStream(src=0).start()

    num=1

    # loop over the frames from the video stream
    while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame
            
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face and their corresponding locations
            for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    
                    if label=="No Mask":
                        sound.play()
                        def load_name(folder):
                            i=0
                            for filename in os.listdir(folder):
                                img = cv2.imread(os.path.join(folder,filename))
                                i=i+1
                                
                            return i+1
                    
                        
                        count=load_name("C:/Users/lenovo/anaconda3/mainproject/violations/face_mask_violations")
                        if num%7==0 or num==3:
                            cv2.imwrite(r"C:\Users\lenovo\anaconda3\mainproject\violations\face_mask_violations\FMask%d.jpg" % count, frame)
                        num+=1

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Face Mask Detection  (Press 'q' to escape)", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                    break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()



def train_model():

    # initialize number of epochs to train for,
   


    EPOCHS=25
    
    DIRECTORY = r"C:\Users\lenovo\anaconda3\mainproject\Mask Dataset"
    CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
    print("[INFO] Loading Dataset...")
    data = []
    labels = []
    print("\nStarting Training...")

# loop over the image paths
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(category)
# convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

# perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    INIT_LR = 1e-4
    BS = 32

# partition the data into training and testing splits 
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# not be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

# compile our model
   
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

# train the head of the network
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

# make predictions on the testing set-testing model
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))

# serialize the model to disk
    model.save("mask_detector.model", save_format="h5")
    print("\n Trained Model successfully saved!\n")

# plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

	


def social_distancing():
    #load the face detection files
    face_model = cv2.CascadeClassifier(r'C:\Users\lenovo\anaconda3\mainproject\face_detector\haarcascade_frontalface_default.xml')
    #load alarm
    mixer.init()
    sound = mixer.Sound(r'C:\Users\lenovo\anaconda3\mainproject\alarm.mp3')
    #starting the videostream
    cap=cv2.VideoCapture(0)
    print("\n[INFO]Starting Video...")
    print("\n Checking Social Distance...\n")
    num=1
    while True:
        status , photo = cap.read()
        frame = photo
        frame = imutils.resize(frame, width=500)
        
        #detecting the faces from the obtained video frame
        face_cor = face_model.detectMultiScale(photo)
        l = len(face_cor)
        photo = cv2.putText(photo, str(len(face_cor))+" Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        stack_x = []
        stack_y = []
        stack_x_print = []
        stack_y_print = []
        global D

        if len(face_cor) == 0:
            pass
        else:
            for i in range(0,len(face_cor)):
                x1 = face_cor[i][0]
                y1 = face_cor[i][1]
                x2 = face_cor[i][0] + face_cor[i][2]
                y2 = face_cor[i][1] + face_cor[i][3]

                #calculating the centre of the detected faces from the frame and appending them to lists
                mid_x = int((x1+x2)/2)
                mid_y = int((y1+y2)/2)
                stack_x.append(mid_x)
                stack_y.append(mid_y)
                stack_x_print.append(mid_x)
                stack_y_print.append(mid_y)
                photo = cv2.circle(photo, (mid_x, mid_y), 3 , [255,0,0] , -1)
                photo = cv2.rectangle(photo , (x1, y1) , (x2,y2) , [0,255,0] , 2)

            if len(face_cor) == 2:
                #calculating the euclidean distance and then drawing a red line between those faces
                D = int(dist.euclidean((stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())))
                photo = cv2.line(photo, (stack_x_print.pop(), stack_y_print.pop()), (stack_x_print.pop(), stack_y_print.pop()), [0,0,255], 2)
            else:
                D = 0

            if D<250 and D!=0:
                #in case of violations drawing line between faces,showing the gap and warning message and invokes the alarm
                photo = cv2.putText(photo, "PLEASE MAINTAIN SOCIAL DISTANCING!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,1, [0,0,255] , 2)
                photo = cv2.putText(photo, str(D/10) + " cm",(300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0) , 2, cv2.LINE_AA)
                sound.play()
                time.sleep(0.05)
                #saving the images of violations to the disk
                def load_name(folder):
                    i=0
                    for filename in os.listdir(folder):
                        img = cv2.imread(os.path.join(folder,filename))
                        i=i+1
                                    
                    return i+1
                        
                            
                count=load_name("C:/Users/lenovo/anaconda3/mainproject/violations/social_distance_violations")
                if num%7==0 or num==3:
                    cv2.imwrite(r"C:\Users\lenovo\anaconda3\mainproject\violations\social_distance_violations\Socialdist%d.jpg" % count, frame)
                num+=1

            cv2.imshow("Social Distance (violation<25cm)         (Press 'q' to exit)", photo)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cap.release()
            break
    
    cv2.destroyAllWindows()


    

def graph_show():
    print("Displaying Graph of training") 
    img = cv2.imread(r"C:\Users\lenovo\anaconda3\mainproject\plot.png")
    winname = "visualisation (Press 'q' to exit)"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 360,30)  # Move it to (360,30)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    

def mviolations_show():
    images = []
    folder=r'C:\Users\lenovo\anaconda3\mainproject\violations\face_mask_violations'
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
   
    if not images:
        print("No images of face mask violations")
    else:
        print("Displaying face mask violations")
        for i in range(0,len(images)):
            pic=images[i]
            winname = "violations (Press any key to see next)"
            cv2.namedWindow(winname)    
            cv2.moveWindow(winname, 300,100)
            cv2.imshow(winname, pic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def sviolations_show():
    images = []
    folder=r'C:\Users\lenovo\anaconda3\mainproject\violations\social_distance_violations'
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
   
    if not images:
        print("No images of Social Distance violations")
    else:
        print("Displaying Social distance violations")
        for i in range(0,len(images)):
            pic=images[i]
            winname = "violations (Press any key to see next)"
            cv2.namedWindow(winname)    
            cv2.moveWindow(winname, 300,100)
            cv2.imshow(winname, pic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    

def prgm_close():
    print("\n\tTerminated\n")
    window.quit()




#code for buttons
trainImg = tk.Button(window, text ="Train Model(Mask detection)",  
command = train_model, fg ="black", bg ="turquoise",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 100, y = 300)

takeImg = tk.Button(window, text ="Face Mask",  
command = face_mask, fg ="black", bg ="turquoise",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
takeImg.place(x = 500, y = 300)

trainImg = tk.Button(window, text ="Social Distance",  
command = social_distancing, fg ="black", bg ="turquoise",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 900, y = 300)

trainImg = tk.Button(window, text ="Training graph",  
command = graph_show, fg ="black", bg ="turquoise",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 100, y = 500)

trainImg = tk.Button(window, text ="Mask Violations",  
command = mviolations_show, fg ="black", bg ="turquoise",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 500, y = 500)

trainImg = tk.Button(window, text ="Distance Violations",  
command = sviolations_show, fg ="black", bg ="turquoise",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 900, y = 500)

trainImg = tk.Button(window, text ="Exit",  
command = prgm_close, fg ="black", bg ="gold",  
width = 15, height = 2, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 1100, y = 620)




window.mainloop() 


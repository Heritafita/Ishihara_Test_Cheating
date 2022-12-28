from preprocess_Image import preprocess_Image
import os
import cv2 
import numpy as np
from keras.models import load_model

#### Initializing the webcam
cap = cv2.VideoCapture(0)

#### loading last trained model
model = load_model("CNN")

#### reading Image & Preprocessing
pi = preprocess_Image()


while True :   
        ret, input_img = cap.read()
        LAB_converted = pi.BGR2LAB(input_img, type = "LAB")
        LAB_converted = cv2.resize(LAB_converted,(500,500))       

        # only keeping the A channel representing the red-green coloring
        _, A, _ = cv2.split(LAB_converted)
        processed_image = pi.highlight_target(A)

        #### reshaping and predict with trained model
        # reshaping to fit
        img = cv2.resize(processed_image,(28,28))
        img = np.reshape(img, (1, 28, 28, 1))

        # scaling
        img = (img - 0) / (255 - 0)

        # make prediction
        pred = model.predict(img)

        # predicting the number for each test
        number = str(pred.argmax())
        result = "Predicted : " + number
        font = cv2.FONT_HERSHEY_COMPLEX
        # Creating zeros matrix 
        zeros = np.zeros(processed_image.shape[:2], dtype="uint8")
        # creating 2 empty channels so that we can overlay the processed image with the original
        three_channel_output = cv2.merge([processed_image,zeros,zeros])
        input = LAB_converted
        #superimpose input and output
        overlayed = cv2.addWeighted(input,0.4,three_channel_output,0,0)
        cv2.putText(overlayed ,result,(15,75),font,1,(127,255,0),1) #text,coordinate,font,size of text,color,thickness of font
        cv2.imshow('Output', overlayed)
        if cv2.waitKey(1) == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()

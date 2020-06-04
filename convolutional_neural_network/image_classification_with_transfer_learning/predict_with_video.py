import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('models/cats_vs_dogs_pretrained_with_fine_tuning.h5')
print('\n-------------------------------------------------------------------\n')
print('\nModel loaded successfully...')

class_names = ['cat', 'dog']
get_class_names = lambda x : class_names[0] if x[0] < 0 else class_names[1]
IMG_SIZE = 160
threshold = 1

# Video capture
cap = cv2.VideoCapture('videos/cat_and_dog.mp4')

# Write the result
# out = cv2.VideoWriter(DIR_PATH + 'result_cat_vs_dog.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,640))

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()

    # Processing frame
    frame_preprocessed = tf.cast(frame, tf.float32)
    frame_preprocessed = (frame/(255/2)) - 1 # rescale the input channels to a range of [-1,1]
    frame_preprocessed = tf.image.resize(frame_preprocessed, (IMG_SIZE, IMG_SIZE)) # Resize the images to a fixed input size

    # Predict
    prediction = model.predict(np.array([frame_preprocessed]))
    predicted_class = get_class_names(prediction)
    if abs(prediction[0][0]) > threshold:
        text = f"It's a {predicted_class}, score : {abs(prediction[0][0]):.2f}"
        print(text)
    else:
        text = f"I'm not sure what it is, score : {abs(prediction[0][0]):.2f}"
        print(text)

    cv2.putText(frame, text, (0,50), cv2.FONT_HERSHEY_PLAIN, 2.5, (255,0,0), 2, cv2.LINE_AA)

    # Write the result
    # out.write(frame)

    # Show result
    cv2.namedWindow("Result", 0)
    cv2.resizeWindow("Result", IMG_SIZE*5,IMG_SIZE*5)
    cv2.imshow("Result", frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

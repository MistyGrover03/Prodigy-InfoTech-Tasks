from keras.models import model_from_json, Sequential
import cv2
import numpy as np

# Load the model structure
with open("signlanguagedetection48x48.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load model weights
model.load_weights("signlanguagedetection48x48.h5")
print("Model loaded successfully.")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

cap = cv2.VideoCapture(0)
label = ['A', 'M', 'N', 'S', 'T', 'blank']

if not cap.isOpened():
    print("Error: Could not open video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (48, 48))
    crop_frame = extract_features(crop_frame)
    
    print("Running model prediction...")
    pred = model.predict(crop_frame)
    print(f"Prediction: {pred}, Max Index: {pred.argmax()}, Label: {label[pred.argmax()]}")
    
    prediction_label = label[pred.argmax()]
    
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("output", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

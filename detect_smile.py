import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"C:\Users\ASUS\Desktop\Smile_Detection-master\Smile_Detection-master\smile_detection_model.h5")

# Load the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect smiles in an image
def detect_smiles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate through the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64))  # Resize to match model input size
        face_roi = face_roi.astype("float") / 255.0  # Normalize pixel values

        # Reshape the ROI to match the input shape expected by the model (add batch dimension)
        face_roi = face_roi.reshape(1, 64, 64, 1)

        # Predict smile probability using the model
        smile_prob = model.predict(face_roi)[0][1]  # Probability of smiling (class 1)

        # Determine the label based on the predicted probability
        label = "Smiling" if smile_prob > 0.5 else "Not Smiling"

        # Draw bounding box and label on the image
        color = (0, 255, 0) if label == "Smiling" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Display the image with detected smiles
    cv2.imshow("Smile Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image to be processed
# Path to the image to be processed
image_path = r"C:\Users\ASUS\Desktop\Smile_Detection-master\Smile_Detection-master\smile.jpeg"

# Call the function to detect smiles in the image
detect_smiles(image_path)

import cv2

# Create a variable body_classifier to assign the CascadeClassifier file
body_classifier = cv2.CascadeClassifier('D:\python\Project 106\haarcascade_fullbody.xml')

# Open the video file
cap = cv2.VideoCapture('D:\python\Project 106\walking.avi.mp4')

while cap.isOpened():
    # Read each frame
    ret, frame = cap.read()
    
    # If there's an issue reading the frame, break out of the loop
    if not ret:
        break
    
    # Convert each frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass each frame to the classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Create a for loop for each x, y, w, h captured in bodies
    for (x, y, w, h) in bodies:
        # Draw a rectangle around a detected area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
    # Display the frame
    cv2.imshow('Pedestrian Detection', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()

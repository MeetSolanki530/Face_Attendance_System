import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import threading

# Helper function to load and encode images
def load_and_encode_image(image_path):
    image = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(image)[0]

# Load and encode images
known_face_encodings = [
    load_and_encode_image(r"Images\bose_subhash.jpeg"),
    load_and_encode_image(r"Images\kartik.jpeg"),
    load_and_encode_image(r"Images\Gandhiji.jpeg"),
    load_and_encode_image(r"Images\Shani.jpeg"),
    load_and_encode_image(r"Images\nehruji.jpg"),
    load_and_encode_image(r"Images\Meet.jpg")
]

# Store Student Names
known_face_names = [
    'Subhash Chandra Bose',
    'Kartik Aaryan',
    'Mahatma Gandhi',
    'Shani Darji',
    'Jawaharlal Nehru',
    'Meet Solanki'
]

# Copy student names to the students variable
students = known_face_names.copy()

# Variables for tracking face locations, encodings, and names
face_locations = []
face_encodings = []
face_names = []

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Get the current date for CSV file naming
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open a CSV file to write attendance records
f = open(current_date + '.csv', 'w', newline='')
csv_writer = csv.writer(f)

# Variables for frame processing
frame_count = 0
process_every_n_frames = 100

def process_frame():
    global face_locations, face_encodings, face_names, students, frame

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H:%M:%S")
            csv_writer.writerow([name, current_time])

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_count % process_every_n_frames == 0:
        threading.Thread(target=process_frame).start()
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale = 1.5
        fontColor = (255,0,0)
        thickness = 3
        lineType  = 2

        cv2.putText(frame, name +' Present', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    cv2.imshow("Attendance System", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
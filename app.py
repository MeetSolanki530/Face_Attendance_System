import face_recognition #show stopper recognise faces with present faces in database
import cv2 #import webcame input and give face recognition
import numpy as np #use for numpy array 
import csv #handke csv files
import os 
from datetime import datetime #get exact date and time


#getting video capture
video_capture = cv2.VideoCapture(0) #0 for default camera we can change depend on our external number of camera

#reading images and encoding it
bose_image = face_recognition.load_image_file(r"Images\bose_subhash.jpeg")
bose_image = cv2.cvtColor(bose_image, cv2.COLOR_BGR2RGB)

bose_encoding = face_recognition.face_encodings(bose_image)[0]

kartik_image = face_recognition.load_image_file(r"Images\kartik.jpeg")
kartik_image = cv2.cvtColor(kartik_image, cv2.COLOR_BGR2RGB)

kartik_encoding = face_recognition.face_encodings(kartik_image)[0]

gandhi_image = face_recognition.load_image_file(r"Images\Gandhiji.jpeg")
gandhi_image = cv2.cvtColor(gandhi_image, cv2.COLOR_BGR2RGB)

gandhi_encoding = face_recognition.face_encodings(gandhi_image)[0]

shani_image = face_recognition.load_image_file(r"Images\Shani.jpeg")
shani_image = cv2.cvtColor(shani_image, cv2.COLOR_BGR2RGB)

shani_encoding = face_recognition.face_encodings(shani_image)[0]

nehru_image = face_recognition.load_image_file(r"Images\nehruji.jpg")
nehru_image = cv2.cvtColor(nehru_image, cv2.COLOR_BGR2RGB)

nehru_encoding = face_recognition.face_encodings(nehru_image)[0]

meet_image = face_recognition.load_image_file(r"Images\Meet.jpg")
meet_image = cv2.cvtColor(meet_image, cv2.COLOR_BGR2RGB)

meet_encoding = face_recognition.face_encodings(meet_image)[0]


#Store Student Face Encodings
known_face_encoding = [
    bose_encoding,
    kartik_encoding,
    gandhi_encoding,
    shani_encoding,
    nehru_encoding,
    meet_encoding
]

#Store Student Names
known_faces_names = [
    'Subhash Chandra Bose',
    'Kartik Aaryan',
    'Mahatma Gandhi',
    'Shani Darji',
    'Jawaharlal Nehru',
    'Meet Solanki'

]


#Copy student name in students variable
students = known_faces_names.copy()

face_locations = [] #used to save face location used to store face from webcam
face_encodings = [] #save encodings
face_names = [] #store name of face if it is in above list of names
s= True

#save date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline='') #csv file name is current date and store their names
inwriter = csv.writer(f)

#created infinite loop
while True:
    ret,frame = video_capture.read() #storing video frames extracting data

    small_frame = cv2.resize(frame,(0,0),fx = 0.25,fy = 0.25)#resizing frames
    rgb_small_frame = small_frame[:,:,::-1] #convert bgr into rgb
    if s: 
        #below two line code for check face is available or not run if face available
        face_locations = face_recognition.face_locations(rgb_small_frame)#if there is face or not in frame
        
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) #store face data from frames

        #in below checking and comparing face from frame
        face_names = []

        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distances) #to get best fit
            
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            #appending face names list
            face_names.append(name) #storing name of face
            
            if name in known_faces_names: #if this is true
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

                if name in students: #if this is true
                    students.remove(name) #removing name because we want save student name multiple times save only time
                    print(students)
                    current_time = now.strftime("%H:%M:%S") #then we will get back current time
                    inwriter.writerow([name,current_time]) #save name with current time
    
    cv2.imshow("Attendance System",frame) #name and frame 
    if cv2.waitKey(1) & 0xFF == ord('q'): #exist video condtion
        break

video_capture.release() 
cv2.destroyAllWindows() #destroy all windows cv2
f.close() #closing csv file


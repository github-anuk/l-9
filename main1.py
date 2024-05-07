import cv2,os,sys,numpy
 
harr_file="C:/Users/Anukriti/Desktop/all my files (anukriti)/python/projects/open cv/l-9/haarcascade_frontalface_default.xml"
datasets = "C:/Users/Anukriti/Desktop/all my files (anukriti)/python/projects/open cv/l-9/Datasets"

(images,lables,names,id)= ([],[],{},0)


for(subdir,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir #fetch the names and save it inside
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath + "/" + filename
            label=id
            images.append(cv2.imread(path,0))
            lables.append(int(label))
        id+=1


(width,height)=(130,100)


#create a numpy array from lists


(images,lables)=[numpy.array(lis) for lis in [images,lables] ]



recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.train(images,lables)
face_cascade = cv2.CascadeClassifier(harr_file)
webcam=cv2.VideoCapture(0)

while True:
    ret,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))
        prediction=recogniser.perdict(face_resize)
        print(prediction)
        cv2.rectangle(im,(x,y), (x+w,y+h),(0,255,0),3)

        if prediction[1]<500:
           cv2.putText(im, '% s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10),
cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))
        else:
          cv2.putText(im, 'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
 
    cv2.imshow('OpenCV', im)
     
    key = cv2.waitKey(10)
    #space key
    if key == 27:
        break
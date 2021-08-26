path = "path of multiple videos to test"
pathlen = len(path) - 5
path_videos = [i for i in glob.glob(path)]
leng = len(path_videos)
print('total videos', leng)
num_samples = 300  
num_frames = 30  
h, w, c = 200, 200, 3 
size = (h,w)
images = []
model = tf.keras.models.load_model('path of saved model - Normal [stage detectition] ', custom_objects={'TCN': TCN})
model2 = tf.keras.models.load_model('path of saved model - body[ detect body N/F].h5', custom_objects={'TCN': TCN})
model3 = tf.keras.models.load_model('path of saved model - crown [ detect body N/F].h5', custom_objects={'TCN': TCN})

font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
thickness = 2

txt_normal = "Normal"

txt_predict_coordinate = (550, 50)
txt_name_coorindinate =  (300,350)
txt_predict_color = (255,255,255)
fail_color = (0, 0, 255)
normal_color = (0,255,0) 

# if the video is cropped, comment the bottom line of code
top, right, bottom, left =  100, 772+10, 370+100, 10   

for i in range(0,leng):
  input_videos = path_videos[i]
  name = input_videos[pathlen:]

  cam = cv2.VideoCapture(input_videos)
  frame_number = cam.get(cv2.CAP_PROP_FRAME_COUNT)
  fps = int(cam.get(cv2.CAP_PROP_FPS))
  h, w = 200, 200
  size = (h,w)
  print("Total number of frames in the video",frame_number)
  print("Frames per second",fps)
  temp = []
  temp2 = []
  video = []
  predicted_values = []
  count_Normal = 0
  count_Fail = 0
  for i in range(int(frame_number)):
    ret, frames = cam.read()

    # if the video is cropped, comment the bottom line of code
    frames = frames[top:bottom, left:right]

    if ret:

      video.append(frames)
      image_list = cv2.resize(frames, (200,200))
      pixels = np.asarray(image_list)
      pixels = pixels.astype('float32')
      pixels /= 255.0
    
      if len(temp) < fps:
        temp.append(pixels)
      else:
        temp2.append(temp)
        
        X_test = np.array(temp2)
        y_predict = model.predict(X_test) # [stage classification]

          #Max Value in three categories 
          #Multi-class
                     
        if y_predict[0][0] > y_predict[0][1] and y_predict[0][0] > y_predict[0][2]: #Category-1

            #If Detected [BODY]
          predict_body = model2.predict(X_test)
          print(predict_body)
          if predict_body < 0.12: #Normal
            print("Normal")
            txt = cv2.putText(video[0],"Body", txt_name_coorindinate, font,fontScale, txt_predict_color, thickness, cv2.LINE_AA)
            video1.append(txt)
            cv2_imshow(txt)
            count_Normal += 1
            
          else:
            print("Failure")
            txt = cv2.putText(video[0],"Body",txt_name_coorindinate, font,fontScale, txt_predict_color, thickness, cv2.LINE_AA)
            cv2_imshow(txt)
            print(predict_body)
            count_Fail +=1
          if count_Fail > 0:
            txt_failure = ("Failure:"+ str (count_Fail))
            txt = cv2.putText(video[0],txt_failure, txt_predict_coordinate, font,fontScale, fail_color, thickness, cv2.LINE_AA)
          cv2_imshow(txt)

          #If Detected [crown]
        elif y_predict[0][1] > y_predict[0][0] and y_predict[0][1] > y_predict[0][2]: #Category-2
          print("Crown")
          predict_crown = model3.predict(X_test)
          print(predict_crown)

          if predict_crown < 0.70: #Normal
            txt = cv2.putText(video[0],"Crown",txt_name_coorindinate, font,fontScale, txt_predict_color, thickness, cv2.LINE_AA)
            cv2_imshow(txt)
          else:
            txt = cv2.putText(video[0],"Crown",txt_name_coorindinate, font,fontScale, txt_predict_color, thickness, cv2.LINE_AA)
            cv2_imshow(txt)
            print("Failure")
            print(predict_crown)
            count_Fail +=1

          if count_Fail > 0:
            txt_failure = ("Failure:"+ str (count_Fail))
            txt = cv2.putText(video[0],txt_failure,txt_predict_coordinate, font,fontScale, fail_color, thickness, cv2.LINE_AA)
          cv2_imshow(txt)

          #If Detected [others]
        elif y_predict[0][2] > y_predict[0][0] and y_predict[0][2] > y_predict[0][1]: #Category-3
          txt = cv2.putText(video[0],"Others",txt_name_coorindinate, font,fontScale, txt_predict_color, thickness, cv2.LINE_AA)
          cv2_imshow(txt)
          print("Other")

        temp2.clear()
        temp.clear()
    video.clear()

import cv2 as cv

import os
from os import listdir
from os.path import isfile, join

def extract(video_path, output_path):

    cap = cv.VideoCapture()
    cap = cv.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")
    count = 1
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        # Display the resulting frame
        prefix = '000'
        img_nm = prefix + str(count)
        img_nm = img_nm[-4:]


        print(img_nm, count)

        # cv.imshow('Frame',frame)
        cv.imwrite(output_path+'/'+img_nm+'.png', frame)

        count=count + 1





        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
          break

      # Break the loop
      else:
        break
    print('Done with ' + video_path + str(count-1) + 'frames found')
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


path = '../DHF1K/video/'
for f in listdir(path):
    image_path = '../DHF1K/images/' + f[:-4]
    # print(image_path)
    # Create directory

    isExist = os.path.exists(image_path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(image_path)
    extract(path + f, image_path)



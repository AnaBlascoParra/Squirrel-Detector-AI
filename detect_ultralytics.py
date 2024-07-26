from PIL import Image
import os


#for root, dirs, files in os.walk("./images", topdown=False):
#    for name in files:
#        file_name = os.path.join(root, name)
#        print(file_name)
#        img = Image.open(file_name)
#        img = img.resize((640,640))
#        img.save(file_name.replace("images", "images_640"))
#
#
#
#exit()


#path_img1 = 'img1.png'
#path_img2 = 'img2.png'
#path_img3 = 'img3.jpg'
#path_img4 = 'img4.jpg'
#
#img1 = cv2.imread(path_img1)
#img2 = cv2.imread(path_img2)
#img3 = cv2.imread(path_img3)
#img4 = cv2.imread(path_img4)
#imgs = [img1, img2, img3, img4]

from ultralytics import YOLO

model = YOLO('yolov8x.pt')

if __name__ == '__main__':
    results = model.train(data='wiwiyolo2.yaml', epochs=50, imgsz=640, verbose=False)

    print(results)

    for root, dirs, files in os.walk("./images", topdown=False):
        for name in files:
            file_name = os.path.join(root, name)
            print("Detect: ", file_name)
            img = Image.open(file_name)
            predicted =  model.predict(img, save=True,imgsz=640,conf=0.5)
















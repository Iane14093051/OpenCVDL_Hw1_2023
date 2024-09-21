import cv2
import math
import numpy as np
import torch
import sys
from PIL import ImageTk, Image
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
from torchsummary import summary
import torchvision.models as models
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLineEdit,
                             QLabel, QPushButton, QMainWindow,  QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from torchvision import models
from torchvision.transforms import v2
from torchvision import transforms
import matplotlib.pyplot as plt


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Gui(QtWidgets.QMainWindow):
    ga_img =[]
    bi_img =[]
    me_img =[]
    sobel_x = [[-1,0,1],
               [-2,0,2],
               [-1,0,1]]

    sobel_y = [[-1,-2,-1],
              [0,0,0],
              [1,2,1]]
    f_name = []

    def __init__(self, model_path):
        super().__init__()

        self.image_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model(model_path)


        self.initUI()

    def initUI(self):


        self.setWindowTitle("Hw1")
        self.setGeometry(200, 200, 1280, 720)
        self.setFixedSize(1280, 720)

        self.area1 = QtWidgets.QGroupBox("1. Image Processing", self)
        self.area1.setFont(QtGui.QFont("Arial", 8))
        self.area1.setGeometry(50, 40, 300, 200)

        self.area2 = QtWidgets.QGroupBox("2. Image Smoothing", self)
        self.area2.setFont(QtGui.QFont("Arial", 8))
        self.area2.setGeometry(50, 340, 300, 300)

        self.area3 = QtWidgets.QGroupBox("3. Image Transformations", self)
        self.area3.setFont(QtGui.QFont("Arial", 8))
        self.area3.setGeometry(350, 40, 300, 300)

        self.area4 = QtWidgets.QGroupBox("4. Transforms", self)
        self.area4.setFont(QtGui.QFont("Arial", 8))
        self.area4.setGeometry(350, 340, 300, 350)

        self.area5 = QtWidgets.QGroupBox("5. VGG19", self)
        self.area5.setFont(QtGui.QFont("Arial", 8))
        self.area5.setGeometry(650, 40, 300, 500)

        color_sep = QPushButton("1.1 Color Separation", self)
        color_sep.setGeometry(60, 60, 160, 50)
        color_sep.clicked.connect(lambda:self.Color(1))

        color_trans = QPushButton("1.2 Color Transformation", self)
        color_trans.setGeometry(60, 120, 160, 50)
        color_trans.clicked.connect(lambda:self.Color(2))

        color_extra = QPushButton("1.3 Color Extraction", self)
        color_extra.setGeometry(60, 180, 160, 50)
        color_extra.clicked.connect(lambda:self.Color(3))


        Gaussian = QPushButton("2.1 Gaussian blur", self)
        Gaussian.setGeometry(60, 360, 160, 50)
        Gaussian.clicked.connect(lambda:self.filter(1))

        Load_Image = QPushButton("Load Image", self)
        Load_Image.setGeometry(60, 250, 200, 50)
        Load_Image.clicked.connect(self.Load_Image)

        
        Bilateral = QPushButton("2.2 Bilateral filter", self)
        Bilateral.setGeometry(60, 420, 160, 50)
        Bilateral.clicked.connect(lambda:self.filter(2))


        Median = QPushButton("2.3 Median filter", self)
        Median.setGeometry(60, 480, 160, 50)
        Median.clicked.connect(lambda:self.filter(3))


        Sobelx = QPushButton("3.1 Sobel X", self)
        Sobelx.setGeometry(360, 60, 160, 50)
        Sobelx.clicked.connect(lambda:self.mask(1))

        Sobely = QPushButton("3.2 Sobel Y", self)
        Sobely.setGeometry(360, 120, 160, 50)
        Sobely.clicked.connect(lambda:self.mask(2))

        Com_thr = QPushButton("3.3 Combination and Threshold", self)
        Com_thr.setGeometry(360, 180, 160, 50)
        Com_thr.clicked.connect(lambda:self.mask(3))

        ang = QPushButton("3.4 Gradient Angle", self)
        ang.setGeometry(360, 240, 160, 50)
        ang.clicked.connect(lambda:self.mask(4))

 

        scaling = QLabel('Scaling:', self)

        rotation = QLabel('Rotation:', self)
        degree = QLabel('deg', self)
        
        tx = QLabel('Tx:', self)
        pix_x = QLabel('pixel', self)
        
        ty = QLabel('Ty:', self)
        pix_y = QLabel('pixel', self)
        
        global edit_s
        edit_s = QLineEdit(self)
        global edit_r
        edit_r = QLineEdit(self)
        global edit_tx
        edit_tx = QLineEdit(self)
        global edit_ty
        edit_ty = QLineEdit(self)




        scaling.setGeometry(360, 420, 160, 50)
        rotation.setGeometry(360, 360, 160, 50)
        degree.setGeometry(600, 360, 160, 50)
        tx.setGeometry(360, 480, 160, 50)
        pix_x.setGeometry(600, 480, 160, 50)        
        ty.setGeometry(360, 540, 160, 50)
        pix_y.setGeometry(600, 540, 160, 50)


        edit_s.setGeometry(420, 420, 160, 50)
        edit_r.setGeometry(420, 360, 160, 50)
        edit_tx.setGeometry(420, 480,160, 50)
        edit_ty.setGeometry(420, 540, 160, 50)


        Trans = QPushButton("4. Transforms", self)
        Trans.setGeometry(420, 600, 160, 50)
        Trans.clicked.connect(self.Trans)


        augment = QPushButton("1. Show Augmented Images", self)
        augment.setGeometry(660, 120, 160, 50)
        augment.clicked.connect(lambda:self.deep(1))


        structure = QPushButton("2. Show Model Structure", self)
        structure.setGeometry(660, 180, 160, 50)
        structure.clicked.connect(lambda:self.deep(2))


        accur_loss = QPushButton("3. Show Accuracy and Loss", self)
        accur_loss.setGeometry(660, 240, 160, 50)
        accur_loss.clicked.connect(lambda:self.deep(3))


        Infe = QPushButton("4. Inference", self)
        Infe.setGeometry(660, 300, 160, 50)
        Infe.clicked.connect(lambda:self.deep(4))

        Load_img = QPushButton("Load Image", self)
        Load_img.setGeometry(660, 60, 160, 50)
        Load_img.clicked.connect(lambda:self.deep(5))
        
        Predict = QLabel('Predict=', self)
        Predict.setGeometry(660, 360, 160, 50)

        Inference_img = QLabel('Inference Image', self)
        Inference_img.setGeometry(680, 430, 160, 50)
        
        self.area6 = QtWidgets.QGroupBox("", self)
        self.area6.setGeometry(660, 400, 128, 128)

        
        self.image_label = QLabel(self)
        self.image_label.setGeometry(780, 60, 128, 128)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(700, 360, 160, 50)
        
        

    def Load_Image(self):

        try:
             file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
             self.f_name = file_name[0]

        except:
            pass

    def Color(self,opt):
        if opt ==1:
            img = cv2.imread(self.f_name)
            b, g, r =cv2.split(img)
            zeros = np.zeros(img.shape[:2], dtype = "uint8")
            merged_r = cv2.merge([zeros,zeros,r])
            merged_g = cv2.merge([zeros,g,zeros])
            merged_b = cv2.merge([b,zeros,zeros])
            
            cv2.imshow("R channel",merged_r)
            cv2.imshow("G channel",merged_g)
            cv2.imshow("B channel",merged_b)
            cv2.waitKey(0)

        elif opt ==2:
            img = cv2.imread(self.f_name)
            b, g, r =cv2.split(img)
            zeros = np.zeros(img.shape[:2], dtype = "uint8")
            merged_r = cv2.merge([zeros,zeros,r])
            merged_g = cv2.merge([zeros,g,zeros])
            merged_b = cv2.merge([b,zeros,zeros])
            merged_bl = cv2.merge([zeros,zeros,zeros])
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             
            h, w, c = img.shape
            dst = np.zeros((h,w), dtype=np.uint8)
            for row in range(h):
                    for col in range(w):
                            b, g, r = np.int32(img[row,col])
                            
                            y = ((b+g+r)%255)/3
                            dst[row, col] = y
            
            
            cv2.imshow("I1",gray)
            cv2.imshow("I2",dst)
            cv2.waitKey(0)

        elif opt==3:
            img = cv2.imread(self.f_name)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_y = cv2.inRange(hsv_img, (11, 43, 25), (34, 255, 255))
            mask_g = cv2.inRange(hsv_img, (35, 43, 25), (99, 255, 255))
            
            mask = cv2.add(mask_y, mask_g)
            mask_bgr= cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_img = cv2.bitwise_not(mask_bgr,img,mask)
            
            
            cv2.imshow("I1",mask)
            cv2.imshow("I2",mask_img)
            cv2.waitKey(0)        
        
    def filter(self,opt):
    
        if opt==1:
            self.ga_img = cv2.imread(self.f_name)
            cv2.namedWindow('I1')
            cv2.createTrackbar('magnitude', 'I1', 1, 5, self.updateGaussian)
            
            
            cv2.imshow('I1',self.ga_img)
           
            
            cv2.waitKey(0)
        elif opt==2:
            self.bi_img = cv2.imread(self.f_name)
            cv2.namedWindow('I2')
            cv2.createTrackbar('magnitude', 'I2', 1, 5, self.updateBilateral)
            
            
            cv2.imshow('I2',self.bi_img)
           
            
            cv2.waitKey(0)
            #cv2.destoryAllWindows()
        elif opt==3:
            self.me_img = cv2.imread(self.f_name)
            cv2.namedWindow('I3')
            cv2.createTrackbar('magnitude', 'I3', 1, 5, self.updateMedian)
            
            
            cv2.imshow('I3',self.me_img)
           
            
            cv2.waitKey(0)
            #cv2.destoryAllWindows()
        
    def updateGaussian(self,x):
        m = cv2.getTrackbarPos('magnitude', 'I1')
        n_ga = self.ga_img
        n_ga = cv2.GaussianBlur(n_ga, (2*m+1, 2*m+1), 0)
        cv2.imshow('I1',n_ga)
            
        
    def updateBilateral(self,x):
        m = cv2.getTrackbarPos('magnitude', 'I2')
        n_bi = self.bi_img
        n_bi = cv2.bilateralFilter(n_bi, 2*m+1,90,90)
        cv2.imshow('I2',n_bi)
       
    def updateMedian(self,x):
        m = cv2.getTrackbarPos('magnitude', 'I3')
        n_me = self.me_img
        n_me = cv2.medianBlur(n_me, 2*m+1)
        cv2.imshow('I3',n_me)
        
    def mask(self,opt):
        if opt==1:
            img = cv2.imread(self.f_name)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            h, w, c = img.shape

            
            dst = np.zeros((h-2,w-2), dtype=np.int32)
            dst2 = np.zeros((h-2,w-2), dtype=np.uint8)
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst[row][col] = abs(self.sobel_x[0][0]*new_gray[row-1][col-1]+self.sobel_x[0][1]*new_gray[row-1][col]+self.sobel_x[0][2]*new_gray[row-1][col+1]+self.sobel_x[1][0]*new_gray[row][col-1]+self.sobel_x[1][1]*new_gray[row][col]+self.sobel_x[1][2]*new_gray[row][col+1]+self.sobel_x[2][0]*new_gray[row+1][col-1]+self.sobel_x[2][1]*new_gray[row+1][col]+self.sobel_x[2][2]*new_gray[row+1][col+1])
                                 
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst[row][col] = abs(dst[row][col])                      
            
            Max=-1000000
            Min=1000000
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if Max < dst[row][col]:
                          Max = dst[row][col]
                         if Min > dst[row][col]:
                          Min = dst[row][col]
                          
                          
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                    
                         dst[row][col] = (dst[row][col]-Min)*255/(Max-Min)
                         
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                       dst2[row][col]=dst[row][col]
                       
            
                         


                   
            cv2.imshow("I1",dst2)
            cv2.waitKey(0)
            #cv2.destoryAllWindows()
        elif opt==2:
            img = cv2.imread(self.f_name)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            h, w, c = img.shape

            
            dst = np.zeros((h-2,w-2), dtype=np.int32)
            dst2 = np.zeros((h-2,w-2), dtype=np.uint8)
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst[row][col] = abs(self.sobel_y[0][0]*new_gray[row-1][col-1]+self.sobel_y[0][1]*new_gray[row-1][col]+self.sobel_y[0][2]*new_gray[row-1][col+1]+self.sobel_y[1][0]*new_gray[row][col-1]+self.sobel_y[1][1]*new_gray[row][col]+self.sobel_y[1][2]*new_gray[row][col+1]+self.sobel_y[2][0]*new_gray[row+1][col-1]+self.sobel_y[2][1]*new_gray[row+1][col]+self.sobel_y[2][2]*new_gray[row+1][col+1])
                                 
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst[row][col] = abs(dst[row][col])                      
            
            Max=-1000000
            Min=1000000
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if Max < dst[row][col]:
                          Max = dst[row][col]
                         if Min > dst[row][col]:
                          Min = dst[row][col]
                          
                          
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                    
                         dst[row][col] = (dst[row][col]-Min)*255/(Max-Min)
                         
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                       dst2[row][col]=dst[row][col]
                       
            
                         


                   
            cv2.imshow("I1",dst2)
            
            cv2.waitKey(0)
            #cv2.destoryAllWindows()  
        elif opt==3:
            img = cv2.imread(self.f_name)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            h, w, c = img.shape

            
            dst = np.zeros((h-2,w-2), dtype=np.int32)
            dst2 = np.zeros((h-2,w-2), dtype=np.int32)
            
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst2[row][col] = self.sobel_x[0][0]*new_gray[row-1][col-1]+self.sobel_x[0][1]*new_gray[row-1][col]+self.sobel_x[0][2]*new_gray[row-1][col+1]+self.sobel_x[1][0]*new_gray[row][col-1]+self.sobel_x[1][1]*new_gray[row][col]+self.sobel_x[1][2]*new_gray[row][col+1]+self.sobel_x[2][0]*new_gray[row+1][col-1]+self.sobel_x[2][1]*new_gray[row+1][col]+self.sobel_x[2][2]*new_gray[row+1][col+1]



            dst3 = np.zeros((h-2,w-2), dtype=np.int32)
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst3[row][col] = self.sobel_y[0][0]*new_gray[row-1][col-1]+self.sobel_y[0][1]*new_gray[row-1][col]+self.sobel_y[0][2]*new_gray[row-1][col+1]+self.sobel_y[1][0]*new_gray[row][col-1]+self.sobel_y[1][1]*new_gray[row][col]+self.sobel_y[1][2]*new_gray[row][col+1]+self.sobel_y[2][0]*new_gray[row+1][col-1]+self.sobel_y[2][1]*new_gray[row+1][col]+self.sobel_y[2][2]*new_gray[row+1][col+1]

                   
            dst4 = np.zeros((h-2,w-2), dtype=np.uint8)   
            dst5 = np.zeros((h-2,w-2), dtype=np.uint8)


            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst[row][col] = (dst3[row][col]**2+dst2[row][col]**2)**0.5
                         
            Max=-1000000
            Min=1000000
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if Max < dst[row][col]:
                          Max = dst[row][col]
                         if Min > dst[row][col]:
                          Min = dst[row][col]        
                      
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                       dst4[row][col]=(dst[row][col]-Min)*255/(Max-Min)
                       if dst4[row][col]<128:
                          dst5[row][col] = 0
                       else:
                          dst5[row][col] = 255 
                   
            cv2.imshow("I1",dst4)
            cv2.imshow("I2",dst5)
            cv2.waitKey(0)
            #cv2.destoryAllWindows()  
        elif opt==4:
            img = cv2.imread(self.f_name)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            h, w, c = img.shape

            
            dst = np.zeros((h-2,w-2), dtype=np.int32)

            dst8 = np.zeros((h-2,w-2), dtype=np.int32)
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst8[row][col] = (self.sobel_x[0][0]*new_gray[row-1][col-1]+self.sobel_x[0][1]*new_gray[row-1][col]+self.sobel_x[0][2]*new_gray[row-1][col+1]+self.sobel_x[1][0]*new_gray[row][col-1]+self.sobel_x[1][1]*new_gray[row][col]+self.sobel_x[1][2]*new_gray[row][col+1]+self.sobel_x[2][0]*new_gray[row+1][col-1]+self.sobel_x[2][1]*new_gray[row+1][col]+self.sobel_x[2][2]*new_gray[row+1][col+1])

                                 
                         


            dst9 = np.zeros((h-2,w-2), dtype=np.int32)
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst9[row][col] = (self.sobel_y[0][0]*new_gray[row-1][col-1]+self.sobel_y[0][1]*new_gray[row-1][col]+self.sobel_y[0][2]*new_gray[row-1][col+1]+self.sobel_y[1][0]*new_gray[row][col-1]+self.sobel_y[1][1]*new_gray[row][col]+self.sobel_y[1][2]*new_gray[row][col+1]+self.sobel_y[2][0]*new_gray[row+1][col-1]+self.sobel_y[2][1]*new_gray[row+1][col]+self.sobel_y[2][2]*new_gray[row+1][col+1])

                                 
                
            dst4 = np.zeros((h-2,w-2), dtype=np.uint8)   


            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         dst[row][col] = (dst8[row][col]**2+dst9[row][col]**2)**0.5
                         
            Max=-1000000
            Min=1000000
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if Max < dst[row][col]:
                          Max = dst[row][col]
                         if Min > dst[row][col]:
                          Min = dst[row][col]        
                      
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                       dst4[row][col]=(dst[row][col]-Min)*255/(Max-Min)
                       
                       
            dst6 = np.zeros((h-2,w-2), dtype=np.uint8)   
            dst7 = np.zeros((h-2,w-2), dtype=np.uint8)
            
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                      if dst8[row][col] == 0 and dst9[row][col] <0:
                        cos = 270
                      elif dst8[row][col] == 0 and dst9[row][col] >0:
                        cos = 90
                      elif dst8[row][col] == 0 and dst9[row][col] ==0:
                        cos = 0
                      else:
                        cos = math.atan2(dst9[row][col],dst8[row][col])
                        cos*=57.3

                     
                      if dst8[row][col]<0 and dst9[row][col]>0:
                          cos+=180
                      if dst8[row][col]<0 and dst9[row][col]<0:
                          cos+=360
                      if dst8[row][col]>0 and dst9[row][col]<0:
                          cos+=180
                      
                      if cos>=120 and cos<=180:
                          dst6[row][col]=255
                      else:
                          dst6[row][col]=0
                          
                      if cos>=210 and cos<=330:
                          dst7[row][col]=255
                      else:
                          dst7[row][col]=0  
              
            dst6 = cv2.bitwise_and(dst4, dst6)
            dst7 = cv2.bitwise_and(dst4, dst7)
            
            
            cv2.imshow("I1",dst6)
            cv2.imshow("I2",dst7)
            cv2.waitKey(0)
            #cv2.destoryAllWindows()
            
    def Trans(self):

        img = cv2.imread(self.f_name)
        cols,rows, c = img.shape


        M = cv2.getRotationMatrix2D((240,200),float(edit_r.text()),float(edit_s.text()))

        ans = cv2.warpAffine(img,M,(rows,cols))
        H = np.float32([[1,0,0],[0,1,0]])

        H[0][2]= float(edit_tx.text())
        H[1][2]= float(edit_ty.text())
        ans = cv2.warpAffine(ans,H,(rows,cols))

        cv2.imshow("I1",ans)
        cv2.waitKey(0)
        

        
        
    def deep(self,opt):
        if opt ==1:
            img_1 = Image.open(r"Q5_1/automobile.png")  
            img_2 = Image.open(r"Q5_1/bird.png")  
            img_3 = Image.open(r"Q5_1/cat.png")  
            img_4 = Image.open(r"Q5_1/deer.png")  
            img_5 = Image.open(r"Q5_1/dog.png")  
            img_6 = Image.open(r"Q5_1/frog.png")  
            img_7 = Image.open(r"Q5_1/horse.png")  
            img_8 = Image.open(r"Q5_1/ship.png")  
            img_9 = Image.open(r"Q5_1/truck.png")  
            H, W = 32, 32
            #transforms = v2.Compose([
    # transforms.RandomHorizontalFlip(),
    #transforms.RandomResizedCrop(size=(224, 224), antialias=True),
     #v2.RandomVerticalFlip(p=0.5),
    #v2.RandomCrop(32, padding=4),
    #v2.ToTensor(),
    #transforms.RandomRotation(30),
   #v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
#])
            transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
   v2.RandomRotation(30)
])
            img_1 = transforms(img_1)
            img_2 = transforms(img_2)
            img_3 = transforms(img_3)
            img_4 = transforms(img_4)
            img_5 = transforms(img_5)
            img_6 = transforms(img_6)
            img_7 = transforms(img_7)
            img_8 = transforms(img_8)
            img_9 = transforms(img_9)
            
            

            plt.subplot(3,3,1)
            plt.title('automobile')
            plt.imshow(img_1)
           
            
            plt.subplot(3,3,2)
            plt.title('bird')
            plt.imshow(img_2)
            
            plt.subplot(3,3,3)
            plt.title('cat')
            plt.imshow(img_3)
            
            plt.subplot(3,3,4)
            plt.title('deer')
            plt.imshow(img_4)
            
            plt.subplot(3,3,5)
            plt.title('dog')
            plt.imshow(img_5)
            
            plt.subplot(3,3,6)
            plt.title('frog')
            plt.imshow(img_6)
            
            plt.subplot(3,3,7)
            plt.title('horse')
            plt.imshow(img_7)
            
            plt.subplot(3,3,8)
            plt.title('ship')
            plt.imshow(img_8)
            
            plt.subplot(3,3,9)
            plt.title('truck')
            plt.imshow(img_9)
            
     
            plt.show()
            cv2.waitKey(0)
        elif opt ==2:
            device = torch.device("cuda")
            vgg19_bn = models.vgg19_bn(num_classes=10)
            vgg19_bn.to(device)
            summary(vgg19_bn, (3, 32, 32))
            cv2.waitKey(0)
        elif opt ==3:
            img = cv2.imread('training_curve.png')
            cv2.imshow("I1",img)
            cv2.waitKey(0)
        elif opt ==4:
            if hasattr(self, 'image_path') and self.model is not None:
                try:
                    transform = v2.Compose([
                        v2.ToTensor(),
                        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                    ])
                    
                    image = Image.open(self.image_path)
                    transformed_image = transform(image)
                    #qimage = QPixmap(self.image_path).toImage()
                    #pil_image = Image.fromqpixmap(qimage)

     

                    #transformed_image = transform(pil_image)


                    #tensor_image = torch.tensor(transformed_image)
                    #image = self.QImageToTensor(QPixmap(self.image_path).toImage())
                    transformed_image = transformed_image.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output = self.model(transformed_image)

                    _, predicted = torch.max(output, 1)

                    self.result_label.setText(f" {classes[predicted.item()]}")

                    # Plot probability distribution using a histogram
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()
                    plt.bar(classes, probabilities[0])
                    plt.xlabel("Class")
                    plt.ylabel("Probability")
                    plt.title("Probability Distribution")
                    plt.show()
                except Exception as e:
                    print(f"Error during inference: {e}")
        elif opt ==5:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff);;All Files (*)", options=options)

            if file_name:
                self.image_path = file_name
                pixmap = QPixmap(file_name)
                pixmap = pixmap.scaled(128, 128)
                self.image_label.setGeometry(660, 400,128,128)
                self.image_label.setPixmap(pixmap)

    def load_model(self, model_path):
        try:
            self.model = models.vgg19_bn(num_classes=10)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading the model: {e}")
        




    def QImageToTensor(self, qimage):
        width, height = qimage.width(), qimage.height()
        ptr = qimage.bits()
        ptr.setsize(3 * width * height)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        arr = arr.transpose(2, 0, 1)  # 调整通道顺序
        tensor = torch.from_numpy(arr).float()
        tensor /= 255.0  # 标准化像素值
        return tensor
    
def main():
    app = QApplication(sys.argv)
    model_path = "best_model.pth"  # 请替换为您的模型文件路径
    mainWindow = Gui(model_path)
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

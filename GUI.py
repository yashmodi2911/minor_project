from tkinter import *
import cv2
from PIL import Image, ImageTk
from keras.models import model_from_json

class app:
    def __init__(self):
        
        #Camera Initialization
        self.cap=cv2.VideoCapture(0)
        
        #frames for Video panel & Filtered panel, will be described in Video_loop()
        self.current_image = None
        self.current_image2 = None
        
        #The open() function opens a file, and returns it as a file object.
        #Syntax- open(file, mode), file -	The path and name of the file
            #"r" - Read - Default value. Opens a file for reading, error if the file does not exist
            #"a" - Append - Opens a file for appending, creates the file if it does not exist
            #"w" - Write - Opens a file for writing, creates the file if it does not exist
            #"x" - Create - Creates the specified file, returns an error if the file exist
            #"t" - Text - Default value. Text mode
            #"b" - Binary - Binary mode (e.g. images)
    
        self.json_file = open(self.directory+"model-bw.json", "r")
        #JSON - https://www.w3schools.com/python/python_json.asp
        self.model_json = self.json_file.read()
        #The read() method returns the specified number of bytes from the file. Default is -1 which means the whole file.
        self.json_file.close()
        #The close() method closes an open file. You should always close your files, in some cases, due to buffering,
        #changes made to a file may not show until you close the file.
        #For More info -https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        self.loaded_model = model_from_json(self.model_json)
        #tf.keras.models.model_from_json - Parses(analyse (a string or text) into logical syntactic components.)
        #a JSON model configuration string and returns a model instance.
        self.loaded_model.load_weights(self.directory+"model-bw.h5")
        #The weights are saved directly from the model using the save_weights() function and later loaded using the
        #symmetrical load_weights() function.
        
        #Main Window Initialization
        self.root=Tk() 
        self.root.title("Sign to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor) # https://stackoverflow.com/questions/111155/how-do-i-handle-the-window-close-event-in-tkinter
        self.root.geometry("900x850+0+0")#0+0 defines position of the window
        
        #Heading
        self.heading = Label(self.root,text = "ASL recognition using CNN",font=("Comic Sans MS",23,"bold"))
        self.heading.place(x=180,y = 5)
        
        #Video Panel For Camera Input
        self.video=Label(self.root)
        self.video.place(x = 135, y = 60, width = 640, height = 640)
        
        #Filtered image Panel
        self.filter = Label(self.root)
        self.filter.place(x = 460, y = 95, width = 310, height = 310)
        
        #Character Panel
        self.charpanel =Label(self.root) # Current SYmbol
        self.charpanel.place(x = 500,y=640)
        #Character text
        self.char = Label(self.root)
        self.char.place(x = 10,y = 640)
        self.char.config(text="Character :",font=("Courier",30,"bold"))
        
        #Word Panel
        self.wordpanel = Label(self.root) 
        self.wordpanel.place(x = 220,y=700)
        #Word text
        self.word = Label(self.root)
        self.word.place(x = 10,y = 700)
        self.word.config(text ="Word :",font=("Courier",30,"bold"))
        
        #Sentence Panel
        self.senpanel = Label(self.root)
        self.senpanel.place(x = 350,y=760)
        self.sent = Label(self.root)
        self.sent.place(x = 10,y = 760)
        self.sent.config(text ="Sentence :",font=("Courier",30,"bold"))
        
        #Defining variables for character, word, sentence.
        self.str=""
        self.word=""
        self.current_symbol="Empty"
        
        #Calling OpenCV function for video capturing
        self.video_loop()
        
    def video_loop(self):
        
        #Video Recording
        ret, frame = self.cap.read()
        
        #ret=True, if camera is working properly (i.e. frame)
        #ret=False, if any error while capturing
        if ret:
            
            img = cv2.flip(frame, 1)
            #Flipping 2-D array
            #flipcode = 0, About x-axis
            #flipcode = 1, about y-axis
            #flipcode = -1, about both
            
            x1 = int(0.5*frame.shape[1])
            
            '''Shape of image is accessed by img.shape. It returns a tuple of number of rows, columns and channels (if image is color)
                img.shape[0]=Height
                img.shape[1]=Width
                img.shape[2]=Channels'''
                
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            #RGBA color values are an extension of RGB color values with an alpha channel - which specifies the opacity of the color.
            #alpha	-Defines the opacity as a number between 0.0 (fully transparent) and 1.0 (fully opaque)
            
            #PIL.Image.fromarray(obj, mode=None)
            '''Creates an image memory from an object exporting the array interface (using the buffer protocol).
               If obj is not contiguous, then the tobytes method is called and frombuffer() is used.
               Returns an image'''
            
            self.current_image = Image.fromarray(img)
            
            #classPIL.ImageTk.PhotoImage(image=None, size=None, **kw)
            '''A Tkinter-compatible photo image. This can be used everywhere Tkinter expects an image object. If the image is
                an RGBA image, pixels having alpha 0 are treated as transparent.The constructor takes either a PIL image, or a
                mode and a size. Alternatively, you can use the file or data options to initialize the photo image object.

            Parameters: 
            image – Either a PIL image, or a mode string. If a mode string is used, a size must also be given.
            size – If the first argument is a mode string, this defines the size of the image.
            file – A filename to load the image from (using Image.open(file)).
            data – An 8-bit string containing image data (as loaded from an image file).'''
            
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            
            #Configuring Video Panel defined in __init__.
            self.video.imgtk = imgtk
            self.video.config(image=imgtk)
            
            img = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            
            # https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-2-adaptive-thresholding/
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            
            # https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/r
            #First argument is the source image, which should be a grayscale image.
            retval, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            #retval is used in Otsu thresholding  image binarization -- https://medium.com/@hbyacademic/otsu-thresholding-4337710dc519
            '''For this, our cv2.threshold() function is used, but pass an extra flag, cv2.THRESH_OTSU. For threshold value, simply
                pass zero. Then the algorithm finds the optimal threshold value and returns you as the second output, retVal.
                If Otsu thresholding is not used, retVal is same as the threshold value you used.'''
            #res is our thresholded image
            self.predict(res) #calling the predict function
            
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            
            #Configuring Filtered Panel
            self.filter.imgtk = imgtk
            self.filter.config(image=imgtk)
            
            #Configuring Character Panel
            self.panel3.config(text=self.current_symbol,font=("Courier",50))
            #Configuring Word Panel
            self.panel4.config(text=self.word,font=("Courier",40))
            #Configuring Sentence Panel
            self.panel5.config(text=self.str,font=("Courier",40))    
        
            self.root.after(50,video_loop())
            '''Tk().after(delay, callback=None) is a method defined for all tkinter widgets. This method simply calls the function
                callback after the given delay in ms. If no function is given, it acts similar to time.sleep
                (but in milliseconds instead of seconds)'''
            #i.e. framerate(fps) will be 0.05 per second
        
        
    def destructor(self):
        print("Closing Application...")
        self.root.destroy() #Destroying Main Window
        self.cap.release() #Releasing Camera
        cv2.destroyAllWindows()
        
    def predict(self,test_image):
        
        test_image = cv2.resize(test_image, (128,128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
        prediction={}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        #LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        #LAYER 2
        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
        	prediction = {}
        	prediction['D'] = result_dru[0][0]
        	prediction['R'] = result_dru[0][1]
        	prediction['U'] = result_dru[0][2]
        	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
        	prediction = {}
        	prediction['D'] = result_tkdi[0][0]
        	prediction['I'] = result_tkdi[0][1]
        	prediction['K'] = result_tkdi[0][2]
        	prediction['T'] = result_tkdi[0][3]
        	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
        	prediction1 = {}
        	prediction1['M'] = result_smn[0][0]
        	prediction1['N'] = result_smn[0][1]
        	prediction1['S'] = result_smn[0][2]
        	prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        	if(prediction1[0][0] == 'S'):
        		self.current_symbol = prediction1[0][0]
        	else:
        		self.current_symbol = prediction[0][0]
        if(self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if(self.ct[self.current_symbol] > 60):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if(len(self.str) > 16):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol    
pba = app()
pba.root.mainloop()
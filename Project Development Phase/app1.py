
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("vgg16-TeaLeaves-Disease-model.h5",compile=False)
                 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')    

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')  

@app.route('/diseases')
def diseases():
    return render_template('diseases.html')   
    
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')   
    


    
                           
@app.route('/predict',methods = ['GET','POST'])
def upload():
    #uplode and saving the image in uplode imagee
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (224,224)) #making the uploded image in into the format wich  the model works
        x = image.img_to_array(img)#convert it into array
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)
        y=model.predict(x)
        preds=np.argmax(y, axis=1)#taking the maximum prob
        #preds = model.predict_classes(x)
        print("prediction",preds)#predictiiingthe result
        index = ['Anthracnose','algal leaf','bird eye spot', 'brown blight','gray light','healthy','red leaf spot','white spot'] 
        text = "The classified Tea leaf Disease is : " + str(index[preds[0]])
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)




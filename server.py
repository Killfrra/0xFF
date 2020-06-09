from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
#from lightning_evaluator import classify
from PIL import Image
import time
import os
from torchvision.transforms.functional import resize
from PIL.ImageOps import autocontrast

basepath = 'docs'

app = Flask(__name__, static_folder = basepath + '/static')

@app.route('/')
def index():
   cache_timeout = app.get_send_file_max_age('index.html')
   return send_from_directory(basepath, 'index.html', cache_timeout=cache_timeout)

classes = [
   'Ubuntu-Regular', 
   'PlayfairDisplay-Regular', 
   'PTAstraSans-Regular', 
   'PT Astra Serif_Regular', 
   'Akrobat-Regular', 
   'Golos Text_Regular', 
   'MontserratAlternates-Regular', 
   'Roboto-Regular', 
   'Gravity-Regular', 
   'Inter-Regular', 
   'SourceSansPro-Regular', 
   'EBGaramond-Regular', 
   'NotoSans-Regular', 
   'LiberationSerif-Regular', 
   'OpenSans-Regular', 
   'Merriweather-Regular', 
   'Arsenal-Regular', 
   'PlayfairDisplaySC-Regular', 
   'IBMPlexSerif-Regular', 
   'NotoSerif-Regular', 
   'PT Root UI_Regular', 
   'Lora-Regular', 
   'LiberationMono-Regular', 
   'Sansation_Regular', 
   'Montserrat-Regular', 
   'Alegreya-Regular', 
   'Spectral-Regular', 
   'IdealistSans-Regular', 
   'LiberationSans-Regular', 
   'Literata-Regular', 
   'PTSans-Regular', 
   'IBMPlexSans-Regular', 
   'FiraSans-Regular', 
   'UbuntuMono-Regular', 
   'Sreda-Regular', 
   'Oswald-Regular', 
   'Phenomena-Regular', 
   'Colus-Regular', 
   'BebasNeue-Regular', 
   'CormorantGaramond-Regular', 
   'NEXT ART_Regular', 
   'UbuntuCondensed-Regular'
]

@app.route('/upload', methods = ['POST'])
def upload_file():
   f = request.files['image']
   try:
      image = Image.open(f)
      uploads_path = 'uploads'
      os.makedirs(uploads_path, exist_ok=True)
      image.save(uploads_path + '/' + str(round(time.time())) + '_' + secure_filename(f.filename), 'jpeg')
      image = autocontrast(image)
      image = resize(image, 127)
      results = [ x for x in range(0, len(classes)) ] # classify(image)
      results = sorted(zip(results, classes, ['#'] * len(classes)), key=lambda x: x[0], reverse=True)
      response = results
   except Exception as e:
      e = repr(e)
      print(f.filename, e)
      response = { 'error': e }
   
   return jsonify(response)
		
if __name__ == '__main__':
    app.run() #debug = True

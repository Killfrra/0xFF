from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from lightning_evaluator import classify
from PIL import Image
import time
import os

basepath = 'docs'

app = Flask(__name__, static_folder = basepath + '/static')

@app.route('/')
def index():
   cache_timeout = app.get_send_file_max_age('index.html')
   return send_from_directory(basepath, 'index.html', cache_timeout=cache_timeout)

classes = [
   'Golos Text_Regular',
   'NotoSerif-Regular',
   'Oswald-Regular',
   'Phenomena-Regular',
   'Sreda-Regular'
]

@app.route('/upload', methods = ['POST'])
def upload_file():
   f = request.files['image']
   try:
      uploads_path = '/tmp/uploads/'
      os.makedirs(uploads_path, exist_ok=True)
      f.save(uploads_path + str(round(time.time())) + '_' + secure_filename(f.filename))
      image = Image.open(f)
      x, y = int(request.form['x']), int(request.form['y'])
      w, h = int(request.form['width']), int(request.form['height'])
      z, w = x + w, y + h
      image = image.crop((x, y, z, w))
      results = classify(image) # [ x for x in range(0, len(classes)) ]
      results = sorted(zip(results, classes, ['#'] * len(classes)), key=lambda x: x[0], reverse=True)
      response = results
   except Exception as e:
      e = repr(e)
      print(f.filename, e)
      response = { 'error': e }
   
   return jsonify(response)
		
if __name__ == '__main__':
    app.run() #debug = True

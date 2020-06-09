from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from eval import classify
from PIL import Image
import time
import os

basepath = 'docs'

app = Flask(__name__, static_folder = basepath + '/static')

@app.route('/')
def index():
   cache_timeout = app.get_send_file_max_age('index.html')
   return send_from_directory(basepath, 'index.html', cache_timeout=cache_timeout)

from fonts import classes

num_classes = len(classes)

@app.route('/upload', methods = ['POST'])
def upload_file():
   f = request.files['image']
   try:
      image = Image.open(f)
      uploads_path = 'uploads'
      os.makedirs(uploads_path, exist_ok=True)
      image.save(uploads_path + '/' + str(round(time.time())) + '_' + secure_filename(f.filename), 'jpeg')
      results = classify(image) # [ x for x in range(0, len(classes)) ] 
      results = list(map(lambda x: classes[x], results))
   except Exception as e:
      e = repr(e)
      print(f.filename, e)
      results = { 'error': 'Произошла ошибка на сервере' }
   
   return jsonify(results)

@app.route('/comment', methods = ['POST'])
def leave_comment():
   json = request.get_json(force=True) #TODO: fix at client side
   print(str(request.headers), json.get('comment'), json.get('error'))
   return 'OK'
		
if __name__ == '__main__':
    app.run() #debug = True

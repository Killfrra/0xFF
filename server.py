from flask import Flask, render_template, request, send_from_directory, jsonify, redirect
from werkzeug.utils import secure_filename
from onnx_eval import classify
from PIL import Image
import time
import os

basepath = 'docs'

app = Flask(__name__, static_folder = basepath + '/static')

@app.route('/')
def index():
   return send_from_directory(basepath, 'index.html')

from fonts import classes

num_classes = len(classes)
uploads_path = 'uploads'
os.makedirs(uploads_path, exist_ok=True)

@app.route('/upload', methods = ['POST'])
def upload_file():
   f = request.files['image']
   try:
      image = Image.open(f)
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
   with open('feedback.txt', 'a') as feedback_file:
      if request.is_json:
         data = request.json
      else:
         data = request.form
      feedback_file.write(f"""
> {round(time.time())} {request.headers.get('X-Real-IP', 'no real ip')}
> {data.get('error', 'no error')}
> {data.get('comment', 'no comment')}
""")
      return redirect('/contact_us#done')
		
if __name__ == '__main__':
    app.run() #debug = True

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from lightning_evaluator import classify
from PIL import Image
import json

app = Flask(__name__)

@app.route('/')
def index():
   return app.send_static_file('index.html')

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
   #try:
   f.save('/tmp/uploads/' + secure_filename(f.filename))
   image = Image.open(f)
   x, y = int(request.form['x']), int(request.form['y'])
   w, h = int(request.form['width']), int(request.form['height'])
   z, w = x + w, y + h
   image = image.crop((x, y, z, w))
   results = classify(image)
   results = sorted(zip(results, classes, ['#'] * len(classes)), key=lambda x: x[0], reverse=True)
   response = results
   #except Exception as e:
   #   print(f.filename, e)
   #   response = { 'error': True }
   
   return json.dumps(response)
		
if __name__ == '__main__':
    app.run(debug = True)

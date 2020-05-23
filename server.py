from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from main import test_on_single_image
from PIL import Image
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')
	
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      image = Image.open(f)
      results = test_on_single_image(image)

      #f.save('/tmp/df_uploads/' + secure_filename(f.filename))
      return render_template('results.html', results=results)
		
if __name__ == '__main__':
    app.run(debug = True)
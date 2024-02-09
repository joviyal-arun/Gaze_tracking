from flask import Flask, render_template, Response
import cv2
import numpy as np
from analytic_process import demo_1
from flask_cors import CORS
import analytic_process

output_nav=None
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('home_page.html')

@app.route('/video_feed')
def video_feed():
    
    global output_nav
    output = demo_1.run_processing()
    output_nav = output
    return Response(output,mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/test')
def test():
    return analytic_process.testing


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000,debug=True)

# FTT Simple webserver serving static files and print posted body to console
# Listen on port 5000
import sys
import os
import flask
from flask import jsonify
from PIL import Image
import cv2
import numpy as np
import io
import time
import json
import base64
from typing import List, Tuple
import argparse
import open3d
from flask_cors import CORS

# Image placeholder size for /
WIDTH = 50
HEIGHT = 40

# Image Page layout for /
TEMPLATE = '''
<!DOCTYPE html>
<html>
<app>
    <title></title>
    <meta charset="utf-8" />
    <style>
body {
    margin: 0;
    background-color: #FFF;
}
.image {
    display: block;
    margin-left: 2em;
    background-color: #EEE;
    box-shadow: 0 0 5px rgba(0,0,0,0.3);
}
img {
    display: block;
}
    </style>
    <script src="https://code.jquery.com/jquery-1.10.2.min.js" charset="utf-8"></script>
    <script src="http://luis-almeida.github.io/unveil/jquery.unveil.min.js" charset="utf-8"></script>
    <script>
$(document).ready(function() {
    $('img').unveil(1000);
});
    </script>
</head>
<body>
    {% for image in images %}
        <a class="image" href="{{ image.src }}" style="width: {{ image.width }}px; height: {{ image.height }}px">
            <img src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" data-src="{{ image.src }}?w={{ image.width }}&amp;h={{ image.height }}" width="{{ image.width }}" height="{{ image.height }}" alt="{{image.src}}" />
        </a>
        <p>{{image.src}}</p>
    {% endfor %}
</body>
'''

def main(args):
    app = flask.Flask(__name__)
    CORS(app)

    @app.route('/help')
    def print_help():
        return flask.jsonify({'available endpoints': ['GET /', 'GET /<file>', 'POST /cout']})


    @app.route('/inpainted', methods=['POST'])
    def save():
        print([k for k in flask.request.files.keys()])
        if 'image' in flask.request.files:
            img = flask.request.files['image']
            img.save('inpainted.png')

        if 'image_empty' in flask.request.files:
            img = flask.request.files['image_empty']
            img.save('inpainted_raw.png')

        if 'layout' in flask.request.files:
            layout = flask.request.files['layout']
            layout.save('dense_layout.png')

        if 'input_panorama' in flask.request.files:
            input_panorama = flask.request.files['input_panorama']
            input_panorama.save('img.png')

        if 'mask' in flask.request.files:
            mask = flask.request.files['mask']
            mask.save('mask.png')

        if 'predicted_mask' in flask.request.files:
            mask = flask.request.files['predicted_mask']
            mask.save('output/mask.png')

        if 'background_image' in flask.request.files:
            mask = flask.request.files['background_image']
            mask.save('output/background_image.png')
            mask_down = Image.open('D:/codes/torchserve/panodr/ImageServer/output/background_image.png')
            mask_down = mask_down.resize((512,256),Image.LINEAR)
            mask_down.save('output/background_image_inp.png')
            mask_down.save('output/background_image_ds.png')

        if 'background_image_orig' in flask.request.files:
            mask = flask.request.files['background_image_orig']
            mask.save('output/background_image.png')
            mask_down = Image.open('D:/codes/torchserve/panodr/ImageServer/output/background_image.png')
            mask_down = mask_down.resize((4096,2048),Image.LINEAR)
            #print("Saved 4096")
            mask_down.save('output/background_image_ds.png')

        if 'layout_one_hot' in flask.request.files:
            mask = flask.request.files['layout_one_hot']
            mask.save('output/layout_one_hot.png')

        

        body = flask.request.get_data()  # binary
        print(f'---\n{body}\n')
        return '' 


    @app.route('/<path:filename>')
    def get_file(filename):
        """get static files from pwd root"""
        try:
            return flask.send_from_directory('.', filename)
        except IOError:
            flask.abort(404)


    @app.route('/')
    def index():
        """List all images with empty thumbnails, show image on click."""
        images = []
        
        for root, dirs, files in os.walk('.'):
            for filename in [os.path.join(root, name) for name in files]:
                if not filename.endswith(('.jpg', '.png', '.tif', '.tiff')):
                    continue
                im = Image.open(filename)
                w, h = im.size
                aspect = 1.0 * w / h
                if aspect > 1.0 * WIDTH / HEIGHT:
                    width = min(w, WIDTH)
                    height = width / aspect
                else:
                    height = min(h, HEIGHT)
                    width = height * aspect
                images.append({
                    'width': int(width),
                    'height': int(height),
                    'src': filename
                })

        return flask.render_template_string(TEMPLATE, **{
            'images': images
        })


    return app.run(debug=True, host='0.0.0.0')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='asdadasda')
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    main(args)
    sys.exit(main(args))

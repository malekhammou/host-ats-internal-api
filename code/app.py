from create_thumbnail import main
from flask import Flask, request,Response
import os
import m3u8_To_MP4
import base64
#from flasgger import Swagger,swag_from
import json
import time


app = Flask(__name__)
#swagger = Swagger(app, template_file='docs/createThumbnail.yml')

@app.route('/', methods=['GET'])
def index():
    return "Welcome to HOST-ATS pipeline API."
    #return redirect(url_for('flasgger.apidocs'))

@app.route('/createThumbnail', methods=['POST'])
#@swag_from('docs/createThumbnail.yml')

def generateThumbnailFromM3U8():
    data = request.get_json()
    print(data)
    video_url = data['video_url']
    video_filename = data['video_filename'] + '.mp4'
    start_time = time.time()
    m3u8_To_MP4.multithread_download(video_url)
    download_time = time.time() - start_time
    os.rename('m3u8_To_MP4.mp4', video_filename)
    input_folder = os.path.join(os.path.dirname(os.getcwd()), 'input')
    os.makedirs(input_folder, exist_ok=True)
    os.replace(video_filename, os.path.join(input_folder, video_filename))
    start_time = time.time()
    main()
    generate_thumbnail_time = time.time() - start_time

    print("Download time",download_time)
    print("Thumbnail generation time",generate_thumbnail_time)
    output_folder = os.path.join(os.path.dirname(os.getcwd()), 'output', video_filename)

    image_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        return Response('No images found', status=404)
    else:
        image_data = []
        for image_file in image_files:
            with open(os.path.join(output_folder, image_file), 'rb') as f:
                image_bytes = f.read()
            image_data.append({
                'name': image_file,
                'content-type': 'image/jpeg',
                'base64': base64.b64encode(image_bytes).decode('utf-8')
            })
        response_data = {'images': image_data,
                         'downloadTime':download_time,
                         'thumbnailGenerationTime':generate_thumbnail_time}
        response = Response(json.dumps(response_data), mimetype='application/json')
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")


from create_thumbnail import main
from flask import Flask, request,Response
import create_thumbnail
import os
import m3u8_To_MP4
import base64
import json
import time
import psutil
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to HOST-ATS pipeline API."
    #return redirect(url_for('flasgger.apidocs'))

@app.route('/createThumbnail', methods=['POST'])

def generateThumbnailFromM3U8():
    #Timing the whole endpoint
    start_time = time.time()
    #read data from request
    data = request.get_json()
    video_url = data['video_url']
    video_filename = data['video_filename'] + '.mp4'
    
    # Start monitoring CPU usage for video download
    time_download_start = time.time()
    m3u8_To_MP4.multithread_download(video_url)
    time_download_end = time.time()
    os.rename('m3u8_To_MP4.mp4', video_filename)
    input_folder = os.path.join(os.path.dirname(os.getcwd()), 'input')
    os.makedirs(input_folder, exist_ok=True)
    os.replace(video_filename, os.path.join(input_folder, video_filename))
    start_time = time.time()
    time_thumbnail_start = start_time
    main()
    time_thumbnail_end = time.time()
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
        thumbnailGenerationTimeCli=create_thumbnail.frame_extraction+create_thumbnail.models_loading+create_thumbnail.logo_detection+create_thumbnail.face_detection+create_thumbnail.closeup_detection+create_thumbnail.blur_detection,create_thumbnail.iq_predicition
        response_data = {'images': image_data,
                         'downloadTime': time_download_end - time_download_start,
                         'thumbnailGenerationTimeServer': time_thumbnail_end - time_thumbnail_start,
                         'thumbnailGenerationTimeCli': sum(thumbnailGenerationTimeCli),
                         'frameExtractionTime':create_thumbnail.frame_extraction,
                         'modelsLoadingTime':create_thumbnail.models_loading,
                         'logoDetectionTime':create_thumbnail.logo_detection,
                         'faceDetectionTime':create_thumbnail.face_detection,
                         'closeupDetectionTime':create_thumbnail.closeup_detection,
                         'blurDetectionTime':create_thumbnail.blur_detection,
                         'iqaTime':create_thumbnail.iq_predicition}
        
    
        response = Response(json.dumps(response_data), mimetype='application/json')
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        
    return response

    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")


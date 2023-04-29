from create_thumbnail import main
from flask import Flask, request,Response
import os
import requests
import subprocess
import m3u8_To_MP4
import base64

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "Welcome to HOST-ATS api."

@app.route('/create_thumbnail', methods=['POST'])
def download_video():
    data = request.get_json()
    video_url = data['video_url']
    video_filename = data['video_filename'] + '.mp4'
    m3u8_To_MP4.multithread_download(video_url)
    os.rename('m3u8_To_MP4.mp4', video_filename)
    input_folder = os.path.join(os.path.dirname(os.getcwd()), 'input')
    os.makedirs(input_folder, exist_ok=True)
    os.replace(video_filename, os.path.join(input_folder, video_filename))
    main()
    # Get images from output folder and send as multipart response
    output_folder = os.path.join(os.path.dirname(os.getcwd()), 'output', video_filename)

    image_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        return Response('No images found', status=404)
    else:
        boundary = 'image_boundary'
        image_data = []
        for image_file in image_files:
            with open(os.path.join(output_folder, image_file), 'rb') as f:
                image_bytes = f.read()
            image_data.append((
                '--' + boundary,
                'Content-Disposition: form-data; name="image"; filename="' + image_file + '"',
                'Content-Type: image/jpeg',
                '',
                base64.b64encode(image_bytes).decode('utf-8'),
                ''
            ))
        response_data = []
        for item in image_data:
            response_data.extend([bytes(line + '\r\n', 'utf-8') for line in item])
        response_data.append(bytes('--' + boundary + '--\r\n', 'utf-8'))
        response_headers = {
            'Content-Type': 'multipart/form-data; boundary=' + boundary,
            'Content-Length': str(sum(len(line) for line in response_data))
        }
        return Response(b''.join(response_data), headers=response_headers)

if __name__ == '__main__':
    app.run(debug=True)

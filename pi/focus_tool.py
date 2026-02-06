#!/usr/bin/env python3
"""
Multi-camera focus adjustment tool
Open http://<pi-ip>:8080 in any browser
"""

from flask import Flask, Response, request
import subprocess
import cv2
import threading

app = Flask(__name__)

CAMERAS = {
    'side': '/dev/camera_side',
    'top': '/dev/camera_top',
    'front': '/dev/camera_front'
}

# Store active camera streams
streams = {}
locks = {name: threading.Lock() for name in CAMERAS}

HTML = '''
<!DOCTYPE html>
<html>
<head>
  <title>Camera Focus Tool</title>
  <style>
    body { font-family: sans-serif; text-align: center; background: #1a1a1a; color: #fff; }
    .camera-box { display: inline-block; margin: 10px; vertical-align: top; }
    .camera-box img { width: 480px; height: 360px; border: 2px solid #444; background: #000; }
    .camera-box.active img { border-color: #0f0; }
    .controls { margin: 10px 0; }
    input[type=range] { width: 200px; }
    button { margin: 5px; padding: 8px 16px; cursor: pointer; }
    h3 { margin: 5px 0; }
    .status { font-size: 12px; color: #888; }
  </style>
</head>
<body>
  <h1>Camera Focus Tool</h1>
  
  <div class="camera-box" id="box_side">
    <h3>SIDE</h3>
    <img id="img_side" src="/stream/side" onerror="this.style.opacity=0.3">
    <div class="controls">
      <button onclick="setAF('side', true)">AF On</button>
      <button onclick="setAF('side', false)">AF Off</button>
      <br>
      <label>Focus: <span id="val_side">200</span></label><br>
      <input type="range" min="1" max="1023" value="200" oninput="setFocus('side', this.value)">
    </div>
    <div class="status" id="status_side">-</div>
  </div>
  
  <div class="camera-box" id="box_top">
    <h3>TOP</h3>
    <img id="img_top" src="/stream/top" onerror="this.style.opacity=0.3">
    <div class="controls">
      <button onclick="setAF('top', true)">AF On</button>
      <button onclick="setAF('top', false)">AF Off</button>
      <br>
      <label>Focus: <span id="val_top">200</span></label><br>
      <input type="range" min="1" max="1023" value="200" oninput="setFocus('top', this.value)">
    </div>
    <div class="status" id="status_top">-</div>
  </div>
  
  <div class="camera-box" id="box_front">
    <h3>FRONT</h3>
    <img id="img_front" src="/stream/front" onerror="this.style.opacity=0.3">
    <div class="controls">
      <button onclick="setAF('front', true)">AF On</button>
      <button onclick="setAF('front', false)">AF Off</button>
      <br>
      <label>Focus: <span id="val_front">200</span></label><br>
      <input type="range" min="1" max="1023" value="200" oninput="setFocus('front', this.value)">
    </div>
    <div class="status" id="status_front">-</div>
  </div>
  
  <br><br>
  <button onclick="setAllAF(false)" style="background:#ff6600">All AF OFF</button>
  <button onclick="captureAll()" style="background:#0066ff">Capture Test Shots</button>
  
  <script>
    function setFocus(cam, val) {
      document.getElementById('val_' + cam).innerText = val;
      fetch('/focus/' + cam + '/' + val).then(r => r.text()).then(t => {
        document.getElementById('status_' + cam).innerText = t;
      });
    }
    
    function setAF(cam, on) {
      fetch('/autofocus/' + cam + '/' + (on ? '1' : '0')).then(r => r.text()).then(t => {
        document.getElementById('status_' + cam).innerText = t;
      });
    }
    
    function setAllAF(on) {
      ['side', 'top', 'front'].forEach(cam => setAF(cam, on));
    }
    
    function captureAll() {
      fetch('/capture').then(r => r.text()).then(t => alert(t));
    }
  </script>
</body>
</html>
'''

def gen_frames(camera_name):
    device = CAMERAS.get(camera_name)
    if not device:
        return
    
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    while True:
        with locks[camera_name]:
            ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return HTML

@app.route('/stream/<camera>')
def stream(camera):
    if camera not in CAMERAS:
        return 'Unknown camera', 404
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/focus/<camera>/<int:val>')
def set_focus(camera, val):
    device = CAMERAS.get(camera)
    if not device:
        return 'Unknown camera', 404
    result = subprocess.run(
        ['v4l2-ctl', '-d', device, '--set-ctrl', f'focus_absolute={val}'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return f'{camera} focus={val}'
    return f'Error: {result.stderr}'

@app.route('/autofocus/<camera>/<int:on>')
def set_af(camera, on):
    device = CAMERAS.get(camera)
    if not device:
        return 'Unknown camera', 404
    result = subprocess.run(
        ['v4l2-ctl', '-d', device, '--set-ctrl', f'focus_automatic_continuous={on}'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return f'{camera} AF={"on" if on else "off"}'
    return f'Error: {result.stderr}'

@app.route('/capture')
def capture_all():
    """Capture test shots from all cameras"""
    results = []
    for name, device in CAMERAS.items():
        outfile = f'/tmp/test_{name}.jpg'
        result = subprocess.run([
            'ffmpeg', '-y', '-f', 'v4l2', '-input_format', 'mjpeg',
            '-video_size', '4656x3496', '-framerate', '10',
            '-i', device, '-frames:v', '1', outfile
        ], capture_output=True, text=True)
        if result.returncode == 0:
            results.append(f'{name}: OK')
        else:
            results.append(f'{name}: FAILED')
    return '\n'.join(results)

if __name__ == '__main__':
    print("Focus Tool starting...")
    print("Open http://<pi-ip>:8080 in your browser")
    print(f"Cameras: {list(CAMERAS.keys())}")
    
    # Disable autofocus on all cameras at startup
    for name, device in CAMERAS.items():
        subprocess.run(['v4l2-ctl', '-d', device, '--set-ctrl', 'focus_automatic_continuous=0'],
                      capture_output=True)
        print(f"  {name}: AF disabled")
    
    app.run(host='0.0.0.0', port=8080, threaded=True)
import os
import base64
import pymeshlab
import trimesh
import pickle
import multiprocessing as mp
import joblib
from io import BytesIO
import json
from tempfile import TemporaryDirectory
import numpy as np
from pathlib import Path
from flask import Flask, flash, request, redirect, url_for, render_template_string, jsonify
from werkzeug.utils import secure_filename
import pyrender

UPLOAD_FOLDER = '/tmp/upload/'
Path(UPLOAD_FOLDER).mkdir(parents=True,
                          exist_ok=True)
ALLOWED_EXTENSIONS = {
    'mp4',
    'mov',
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def compute_pose(vid_path: str,
                 out_path: str,
                 focal: float,
                 port: int = 8001):
    from xmlrpc.client import ServerProxy
    predictor = ServerProxy(F'http://localhost:{port}')
    # vid_path: str = './examples/IMG_9732.mov'
    # out_path: str = '/tmp/docker/IMG_9732_v2/'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        out = predictor.hand(str(vid_path),
                             str(out_path),
                             float(focal))
        if out == 'ok':
            with open(str(out_path), 'rb') as fp:
                data = pickle.load(fp)
            kpt = [(datum['pred_keypoints_3d'].tolist()
                   if datum is not None
                   else [])
                   for datum in data]
            cam = [(datum['pred_cam_t_full'].tolist()
                    if datum is not None
                    else [])
                   for datum in data]
            rgt = [(datum['is_rights'].tolist()
                    if datum is not None
                    else [])
                   for datum in data]
            # data = json.dumps(dict(kpt=kpt, cam=cam))
            data = dict(kpt=kpt, cam=cam, rgt=rgt)
            return (True, data)
    except Exception as e:
        # return (False, F'{e}')
        raise
        return (False, F'{e}')
    return (False, 'Unknown Failure')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        focal = request.form['focal']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # save
            with TemporaryDirectory(dir=app.config['UPLOAD_FOLDER']) as tmpdir:
                tmpdir = Path(tmpdir)
                filename = secure_filename(F'{tmpdir}/{file.filename}')
                file.save(tmpdir / filename)
                result = compute_pose(str(tmpdir / filename),
                                      str(tmpdir / 'out.pkl'),
                                      focal)
                (suc, out) = result
                return jsonify(dict(
                    success = suc,
                    **out))

                msg = 'Success'
                data = ''
                if not suc:
                    msg = out
                else:
                    data = out
            return render_template_string(
                '''<!doctype html>
                <head>
                    <style>
                    </style>
                </head>

                <body>
                    <p> Success = {{ success }} {{ message }} </p>
                    <p> {{ data }} </p>
                </body>
                    ''',
                success=suc,
                data=data,
                message=msg)

    return '''
    <!doctype html>
    <title>Video</title>
    <h1>Video To Hand Trajectory</h1>
    <form method=post enctype=multipart/form-data>
      <label for="file">Video File</label>
      <input type=file name=file>
      <label for="file">Focal Length</label>
      <input type=number name=focal>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.secret_key = 'im2-demo'
    app.run(host='0.0.0.0', port=5000, debug=False)

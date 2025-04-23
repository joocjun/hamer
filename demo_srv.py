#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass
import pickle
from loguru import logger

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from xmlrpc.client import Binary
import cv2
import numpy as np 
from demo_live import LiveHAMERPipeline

import json
import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


@dataclass
class Config:
    cfg_file: str = 'configs/yamls/demo.yaml'
    device: str = 'cuda:4'
    host: str = '0.0.0.0'
    port: int = 8001
    hamer: LiveHAMERPipeline.Config = LiveHAMERPipeline.Config(
        body_detector='regnety'
    )

def main(cfg: Config):
    pipe = LiveHAMERPipeline(cfg.hamer,
                             'cuda:0',)
    
    def hand(vid_path: str,
             out_path: str,
             focal_length: Optional[float] = None
             ):
        # for i, img_path in enumerate(img_paths):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return F'Failed to load {vid_path}'

        outs = []
        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag:
                break
            out = pipe(img,
                       render_prefix=None,
                       focal_length=focal_length)
            outs.append(out)

        with open(out_path, 'wb') as fp:
            pickle.dump(outs, fp)

        return 'ok'
    
    @logging_time
    def hand_img(img_data: Binary,
                 out_path: str,
                 focal_length: Optional[float] = None):
        

        img = np.frombuffer(img_data.data, dtype=np.uint8)
        img = img.reshape(480, 640, 3)
        out = pipe(img, render_prefix=None, focal_length=focal_length)
        
        if out is None: 
            return 'None'
        else: 
            return json.dumps(
                dict(
                    pred_keypoints_3d=out['pred_keypoints_3d'].tolist(),
                    pred_cam_t_full=out['pred_cam_t_full'].tolist(),
                    pred_vertices=out['pred_vertices'].tolist(),
                    is_rights=out['is_rights'].tolist()
                )
            )

    # Create server
    print('binding to', cfg.host, cfg.port)
    with SimpleXMLRPCServer((cfg.host, cfg.port),
                            requestHandler=RequestHandler) as server:
        server.register_introspection_functions()
        # Register pow() function; this will use the value of
        # pow.__name__ as the name, which is just 'pow'.
        server.register_function(hand, 'hand')
        server.register_function(hand_img, 'hand_img')
        # Run the server's main loop
        server.serve_forever()

    
    logger.info('Done !')


if __name__ == '__main__':
    main(Config())

#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass
import pickle
from loguru import logger

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import cv2

from demo_live import LiveHAMERPipeline


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


@dataclass
class Config:
    cfg_file: str = 'configs/yamls/demo.yaml'
    device: str = 'cuda:0'
    port: int = 8001
    hamer: LiveHAMERPipeline.Config = LiveHAMERPipeline.Config(
        body_detector='regnety'
    )


def main(cfg: Config):
    pipe = LiveHAMERPipeline(cfg.hamer,
                             cfg.device)

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

    def hand_img(img_path: str,
                 out_path: str,
                 focal_length: Optional[float] = None):
        img = cv2.imread(img_path)
        out = pipe(img, render_prefix=None, focal_length=focal_length)
        with open(out_path, 'wb') as fp:
            pickle.dump(out, fp)
        return 'ok'

    # Create server
    with SimpleXMLRPCServer(('localhost', cfg.port),
                            requestHandler=RequestHandler) as server:
        server.register_introspection_functions()
        # Register pow() function; this will use the value of
        # pow.__name__ as the name, which is just 'pow'.
        server.register_function(hand, 'hand')
        server.register_function(hand_img, 'hand_img')
        # Run the server's main loop
        server.serve_forever()

    print()
    logger.info('Done !')


if __name__ == '__main__':
    main(Config())

import os
import cv2
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

class VideoTracker(object):
    '''
    Master class to bind all the detection componenets together
    '''
    def __init__(self, cfg, args):
        '''
        Constructor: Throw warning of cuda supported GPU is not found
        '''
        self.id_tracker = dict()
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            print("Running in cpu mode!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names


    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        

    def run(self):
        '''
        Perform detection, draw bounding boxes and track the ids
        '''
        idx_frame = 0
        while self.vdo.grab(): 
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                    identities = identities.reshape(len(identities),1)

                    #Save the tracking data for the detected ids making sure their x and y co-ordinates match
                    for i in range(len(identities)):
                        xy =  bbox_xyxy[i].reshape(4,1)
                        if identities[i].item() not in self.id_tracker:
                            
                            self.id_tracker[ identities[i].item() ] = list([[np.array(xy[0]) , np.array(xy[1])],identities[i]])
                        else:
                            self.id_tracker[ identities[i].item() ][0][0] = np.append(self.id_tracker[ identities[i].item() ][0][0],xy[0])
                            self.id_tracker[ identities[i].item() ][0][1] = np.append(self.id_tracker[ identities[i].item() ][0][1],xy[1])



            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                #When on video display mode, the running program will quit when 'q' is entered
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if self.args.save_path:
                        self.writer.write(ori_im)
                    break
            
            if self.args.save_path:
                self.writer.write(ori_im)

    def plot_tracked_ids_scatter(self):
        '''
        Perform scatter plot to track the ids
        '''
        plt.figure()
        for id in self.id_tracker:
            ids = self.id_tracker[id]

            plt.scatter(ids[0][0],ids[0][1],label=ids[1])
        plt.legend(loc='upper right')
        plt.savefig('scatter_plot.png')

        plt.show()

    def plot_tracked_ids_plot(self):
        '''
        Perform line plot to track the ids
        '''
        plt.figure()
        for id in self.id_tracker:
            ids = self.id_tracker[id]
            plt.plot(ids[0][0],ids[0][1],label=ids[1])
        plt.legend(loc='upper right')
        plt.savefig('line_plot.png')
        plt.show()
            
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
        vdo_trk.plot_tracked_ids_scatter()
        vdo_trk.plot_tracked_ids_plot()

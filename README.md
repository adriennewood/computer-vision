
**# Studying social interactions in crowded spaces using computer vision**

---

This deep learning neural network identifies people and assigns IDs to track their movements in a crowded room.  Finally, their individual trajectories are mapped on a 2D plane.

YOLOv3 allows for real-time detection, without loss of too much accuracy.  Support for a smaller model was added through Tiny-YOLO, for execution on systems with lesser resources.  Simple Online Realtime Tracking (SORT) was used to assign IDs. It uses Kalman Filters in the backend to uniquely identify objects.  Occlusion led to the problem of inconsistent ID tracking. To tackle this, Deep SORT was used. It computes features for bounding boxes and uses the similarity between features to factor into tracking logic.  Hyperparameters of the open source library were adjusted to suit our use case.

Technologies and tools used: Convolutional Neural Networks, Detector Model (YOLOv3, Tiny-YOLO), Darknet, Pytorch, OpenCV, Matplotlib.


**## Getting started**

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Download YOLOv3 parameters
```
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

3. Download deepsort parameters ckpt.t7
```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from 
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
``` 

4. Copy the video to be detected in the folder containing the code. Or you can pass the full directory as commandline argument.

5. Run demo
```
usage: python yolov3_deepsort.py VIDEO_PATH
                                [--help] 
                                [--frame_interval FRAME_INTERVAL]
                                [--config_detection CONFIG_DETECTION]
                                [--config_deepsort CONFIG_DEEPSORT]
                                [--ignore_display]
                                [--display_width DISPLAY_WIDTH]
                                [--display_height DISPLAY_HEIGHT]
                                [--save_path SAVE_PATH]          
                                [--cpu]          

# YOLOv3 + deepsort 
(This uses the extended YOLOv3 network for high accuracy. Make sure you have a powerful GPU for running this). 
The output of the video with bounding boxes is stored in the demo folder, if not specified otherwise. 
When the video is run on detection mode, you can press 'q' to quit the detection and directly move to the mapping.

python detector.py [VIDEO_PATH]
example=> python detector.py Video.mp4

# YOLOv3_tiny + deepsort (This implementation increases speed, but compromises on accuracy)
python detector.py [VIDEO_PATH] --config_detection ./configs/yolov3_tiny.yaml
```

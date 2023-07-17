import sys
import os
sys.path.append("/module/src/ByteTrack")

import torch
from utils.general import non_max_suppression

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
# from supervision.tools.detections import BoxAnnotator
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
# from tracking_utils import Detection

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

from typing import List
import numpy as np
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.augmentations import letterbox
from tqdm import tqdm
import pandas as pd 

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.1
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids




def track(source_video_path):
    device = ''
    root_dir = os.getcwd()
    root_dir = os.path.join(root_dir, 'src')
    weights = 'runs/train/bowen-run-27-new/weights/best.pt'
    weights = os.path.join(root_dir, weights)
    data = 'data/mare.yaml'
    data = os.path.join(root_dir, data)
    device = select_device(device)

    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (1088, 1920)
    model.warmup(imgsz=(1, 3, *imgsz))

    CLASS_NAMES_DICT = model.model.names
    CLASS_ID = [0,1,2,3,4,5,6,7,8,9]
    HOME = root_dir
    # source_video_path = f"{HOME}/examples/example_more.mp4"
    # target_video_path = f"{HOME}/examples/output/example_more.mp4"
    target_folder_path = f'{HOME}/examples/output/'
    # source_video_path = f"{HOME}/examples/test_video.mp4"
    # source_video_path = f"{HOME}/examples/test_video.mp4"
    _, video_name = os.path.split(source_video_path)
    video_name_no_extension = video_name.split('.')[0]
    video_name = video_name_no_extension + '_output.mp4'
    target_video_path = os.path.join(target_folder_path, video_name)

    CONF_THRES = 0.1
    IOU_THRES = 0.2
    MAX_DET = 1000
    LINE_START = Point(5, 700)
    LINE_END = Point(1920-5, 700)
    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(source_video_path)
    # create frame generator
    generator = get_video_frames_generator(source_video_path)
    # create LineCounter instance
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    
    # open target video file
    frame_num = 0
    with VideoSink(target_video_path, video_info) as sink:
        # loop over video frames
        for frame0 in tqdm(generator, total=video_info.total_frames):
            
            # model prediction on single frame and conversion to supervision Detections
            frame = frame0 / 255
            frame = letterbox(frame, imgsz, stride=32, auto=True)[0]
            # (1088 x 1920 x 3) -> (3 x 1088 x 1920)
            frame = frame.transpose((2, 0, 1))
            frame = np.ascontiguousarray(frame)
            frame = torch.from_numpy(frame).to(device).float()
            frame = frame[None] # expand for batch dim
            # model prediction on single frame and conversion to supervision Detections
            pred0 = model(frame, augment=None, visualize=False)

            # pred = non_max_suppression(pred0, conf_thres, iou_thres, max_det=max_det)
            pred = non_max_suppression(pred0, CONF_THRES, IOU_THRES, None, False, max_det=MAX_DET)
            det = pred[0].cpu().numpy()

            txt_path = os.path.join(target_folder_path, video_name_no_extension) + '.txt'  # im.txt

            xyxy = det[:,:4]
            confidence = det[:,4]
            class_id = det[:,5].astype(int)
            # for i in range(len(xyxy)):
            #     line = (frame_num, class_id[i], *xyxy[i])  # label format
            #     with open(txt_path, 'a') as f:
            #         f.write(('%g ' * len(line)).rstrip() % line + '\n')


            detections = Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame0.shape,
                img_size=frame0.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            # updating line counter
            line_counter.update(detections=detections)
            # annotate and display frame
            frame0 = box_annotator.annotate(frame=frame0, detections=detections, labels=labels)
            line_annotator.annotate(frame=frame0, line_counter=line_counter)
            sink.write_frame(frame0)
            frame_num += 1

            # save detection and tracking to files fomatting as below:
            # frame_num, class_id, tracker_id, *box_xyxy (x_min y_min x_max y_max), confidence
            for box_xyxy, confidence, class_id, tracker_id in detections:
                box_xyxy = box_xyxy.astype(int)
                line = (frame_num, class_id, tracker_id, *box_xyxy, confidence)
                line_str = ('%g ' * len(line)).rstrip() % line + '\n'
                with open(txt_path, 'a') as f:
                    f.write(line_str)
    # breakpoint()
    # print(line_counter.class_dict)
    name_count_dict = {}
    name_count_dict['Total'] = 0
    for i in line_counter.class_dict: 
        name_count_dict[names[i]] = line_counter.class_dict[i]
        name_count_dict['Total'] += line_counter.class_dict[i]
    
    df = pd.Series(name_count_dict)
    hdf_name = f'{video_name_no_extension}_count.hdf5'
    hdf_path = os.path.join(f'{HOME}/examples/output/', hdf_name)
    df.to_hdf(hdf_path, key='counts', format='table')

    return target_video_path, hdf_path, txt_path
if __name__ == "__main__":
    input_path = '/home/bowen68/projects/bisque/Modules/SpeciesTrackerAndCounter/src/examples/example_more.mp4'
    track(input_path)
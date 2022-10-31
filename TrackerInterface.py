

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# start adding in necessary modules for obj det and trackign
import sys

sys.path.append('./Yolov5_StrongSORT_OSNet/')
sys.path.append('./Yolov5_StrongSORT_OSNet/yolov5/')
sys.path.append('./Yolov5_StrongSORT_OSNet/trackers/strong_sort/')
sys.path.append('./Yolov5_StrongSORT_OSNet/trackers/strongsort/deep/reid/torchreid')

from yolov5.models.common import DetectMultiBackend

from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker


'''
	Interfacing with the Object Detection + Tracking
	Software in order to utilize the tracker system for our own purposes.

	In a way, it's tracker.py from the repo, but moved over to a class for general
	work.

	Features:
		- Supports using CUDA in order to perform inference.
		- using yolov5.pt model for inference
		- using uses osnet.pt model for StrongSort
		- stores all meta information regarding detection / tracking
		- can draw on image, the detected stuff

'''
class Yolov5StrongSort_Interface:
	'''
		Parameters
		-----------
			Image Size: 640x480
			conf threshold = 0.25
			iou_threshold = 0.45
			cuda device = defaulted 0
			max detections = 1000
			video sources = 1 # either video or webcam stream
	'''
	def __init__(self, yolo_weights, tracker_weights):
		'''
			yolo_weights and tracker_weights
			--------------------------------
				string of .pt files that go to respective trained models
		'''
		# decide whether to use cpu or gpu
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		self.model = DetectMultiBackend(yolo_weights, device=self.device, \
											dnn=False, data=None, fp16=False)

		self.model_stride = self.model.stride
		self.model_names = self.model.names
		self.model_pt = self.model.pt

		self.model.warmup(1) # warmup yolov5
		self.tracker = create_tracker('strongsort', tracker_weights, self.device, half=False, research_modified=True)
		self.tracker.model.warmup()


	@torch.no_grad()
	def detect(self, im, visual=False):
		'''
			Paramters:
			----------
				im: a numpy wxh array
			Returns:
			--------
				returns resulting prints and image
			Description:
			------------
				Performs the detection and tracking of the detector + tracker
		'''


		# padded resize from LoadStreams
		im_resized = letterbox(im, 640, stride=self.model_stride, auto=True)[0] 

		#add batch dimension
		im = np.ascontiguousarray(im[np.newaxis, :])
		im_resized = im_resized[np.newaxis, :]

		im_resized = im_resized[..., ::-1].transpose((0, 3, 1, 2))
		im_resized = np.ascontiguousarray(im_resized)

		im_t = torch.from_numpy(im_resized).to(self.device).float()
		im_t /= 255.0 # normalize the image

		pred = self.model(im_t, augment=False, visualize=False)

		# params: (prediction, confidence threshold, iou threshold, classes, agnostic, ... 
		#			max detection)
		pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

		outputs = []

		im = im[0].copy()
		for i,det in enumerate(pred):

			annotator = Annotator(im, line_width=2, pil = not ascii)

			# if there are detection
			if (det is not None) and len(det):
				det[:, :4] = scale_coords(im_t.shape[2:], det[:, :4], im.shape).round()


				# setup for tracking update
				xywhs = xyxy2xywh(det[:,0:4])
				confs = det[:,4]
				clss = det[:,5]

				output = self.tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im)
				outputs.append(output)

				# go through each update
				# if visual is turned on, use output for annotation
				if len(output) > 0 and visual:
					for j, (ot, conf) in enumerate(zip(output, confs)):

						bboxes = ot[0:4]
						id = ot[4]
						cls = ot[5]
						c = int(cls)
						id = int(id)

						label = f'{id} {self.model_names[c]} {conf:.2f}'
						annotator.box_label(bboxes, label, color=colors(c,True))

			#set im to annotation if it is turned on
			im = annotator.result()

		return outputs, im
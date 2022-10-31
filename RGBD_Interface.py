import cv2
import pyrealsense2 as intel
import numpy as np
import logging
import threading

'''
	Uses the pyrealsense2 camera library to access
	the Intel Realsense D455 Camera.

	Features:
		- Opens a thread that runs concurrently alongside the main program.
		- Gives Point Clouds per Frame.
		- Gives Depth Frame.
		- Gives RGBD Frame.

'''
class RGBD_Interface(threading.Thread):
	def __init__(self, visual=False):
		threading.Thread.__init__(self)

		logging.debug('Setting up Intel Camera')

		self.visual = visual
		# setting up intel realsense objects

		# pyrealsense2 default config
		self.config = intel.config()
		# pyrealsense2 pipeline object
		self.pipeline = intel.pipeline()

		# pipeline objects
		pipeline_wrapper = intel.pipeline_wrapper(self.pipeline)
		pipeline_profile = self.config.resolve(pipeline_wrapper)

		# device object + device info
		self.device = pipeline_profile.get_device()
		device_product_line = str(self.device.get_info(intel.camera_info.product_line))
		logging.debug(f'Device Product Line: {device_product_line}')

		
		# enabling config stream
		self.config.enable_stream(intel.stream.depth, 640, 480, intel.format.z16, 30)
		self.config.enable_stream(intel.stream.color, 640, 480, intel.format.bgr8, 30)
		logging.debug('Enabled Device Config for streaming')

		# Initializing camera stream
		self.profile = self.pipeline.start(self.config)
		logging.debug('Started Streaming.')

		# setting up RGB + D aligning frames
		align_to = intel.stream.color
		self.align = intel.align(align_to)

		# setting up depth objects
		self.depth_sensor = self.profile.get_device().first_depth_sensor()
		self.depth_scale = self.depth_sensor.get_depth_scale()

		self.pc = intel.pointcloud()
		# these are the attributes that other code will access (the frames)
		self.depth_frame = None
		self.color_frame = None
		self.depth_frame_visual = None
		self.point_clouds = None


	# all of these methods are copied so as to ensure that no code
	# that accesses these frames will write to the frame which causes
	# a synchronization issue
	def get_bgr(self):
		return self.color_frame.copy()
	def get_depth(self):
		return self.depth_frame.copy()
	def get_depth_visual(self):
		if self.depth_frame_visual is None: 
			return None
		return self.depth_frame_visual.copy()
	def get_point_clouds(self):
		return self.point_clouds.copy()

	def get_stream(self):
		return (self.get_bgr(), self.get_depth(), \
				self.get_point_clouds(), self.get_depth_visual())

	def run(self):
		'''
			once thread.start() is called, this method will startup.
			which is why it should be a while loop.

			Threading the camera is not meant for multi-threading, but instead
			as a way to ensure that the camera interface code does not 
			inhibit the main code.
		'''
		try:
			while 1:
				frames = self.pipeline.wait_for_frames()
				aligned_frames = self.align.process(frames)

				depth_frame_obj = aligned_frames.get_depth_frame()
				color_frame_obj = aligned_frames.get_color_frame()

				if not depth_frame_obj or not color_frame_obj:
					logging.debug("ERROR: Depth or Color Frame is not valid.\n")
					continue

				self.depth_frame = np.asanyarray(depth_frame_obj.get_data())
				self.color_frame = np.asanyarray(color_frame_obj.get_data())

				self.pc.map_to(color_frame_obj)
				points = self.pc.calculate(depth_frame_obj)
				self.point_clouds = np.asanyarray(points.get_vertices())

				# if visual is enabled, then get depthmap colorframe
				if self.visual:
					self.depth_frame_visual = cv2.applyColorMap(\
						cv2.convertScaleAbs(self.depth_frame,alpha=0.07), cv2.COLORMAP_JET)
				else: 
					self.depth_frame_visual = None


		except (KeyboardInterrupt, SystemExit):
			# in case anything goes wrong, stop the pipeline.
			self.pipeline.stop()
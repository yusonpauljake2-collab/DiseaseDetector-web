import os
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO


class YoloDiseaseDetector:
	"""Wrapper around Ultralytics YOLO for calamansi disease detection."""

	def __init__(self, model_path: str, device: Optional[str] = None):
		if not os.path.exists(model_path):
			raise FileNotFoundError(f"Model file not found at '{model_path}'. Place your model file in the project root or provide a valid path.")
		self.model = YOLO(model_path)
		if device is not None:
			# Set device if provided (e.g., 'cpu' or '0' for CUDA GPU 0)
			self.model.to(device)

	def predict_image(self, image: Image.Image, conf: float = 0.25, iou: float = 0.50, imgsz: int = 640) -> Tuple[Image.Image, List[Dict]]:
		"""
		Run prediction on a PIL Image and return an annotated PIL Image and detections list.
		Detections list has dicts: {class_id, class_name, confidence, bbox[x1,y1,x2,y2], detection_id}
		"""
		# Ensure RGB then convert to BGR ndarray for Ultralytics
		if image.mode != "RGB":
			image = image.convert("RGB")
		rgb = np.array(image)
		bgr = rgb[:, :, ::-1]

		results = self.model.predict(source=bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
		if not results:
			return image, []

		res = results[0]
		
		# Use original YOLO plot but remove ID numbers from labels
		import cv2
		annotated = res.plot()  # Get original YOLO styling
		annotated_bgr = annotated.copy()
		
		detections: List[Dict] = []
		if res.boxes is not None and len(res.boxes) > 0:
			classes = res.names
			for i, box in enumerate(res.boxes, 1):
				cls_id = int(box.cls.item())
				conf_score = float(box.conf.item())
				x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
				class_name = classes.get(cls_id, str(cls_id)) if isinstance(classes, dict) else str(cls_id)
				
				detections.append(
					{
						"detection_id": i,
						"class_id": cls_id,
						"class_name": class_name,
						"confidence": conf_score,
						"bbox": [x1, y1, x2, y2],
					}
				)
		
		annotated_pil = Image.fromarray(annotated_bgr[:, :, ::-1])
		return annotated_pil, detections

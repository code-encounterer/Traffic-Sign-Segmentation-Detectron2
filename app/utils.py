import cv2
import numpy as np
import os
from PIL import Image
import io
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

def perform_instance_segmentation(image, output_format='png'):
    
    cfg = get_cfg()
    cfg.merge_from_file("config.yml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = os.path.join("models", "model_final.pth")

    predictor = DefaultPredictor(cfg)

    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    outputs = predictor(img[..., ::-1])
    
    metadata = MetadataCatalog.get("traffic_sign_train")
    metadata.thing_classes = ['prohibitory', 'danger', 'mandatory', 'other']
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("traffic_sign_train"), scale=2)

    result_image = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

    result_array = np.asarray(result_image)

    pil_image = Image.fromarray(result_array)
    pil_image = pil_image.resize((680,400))  
    img_buffer = io.BytesIO()

    pil_image.save(img_buffer, format=output_format.upper())

    encoded_image = img_buffer.getvalue()
    return encoded_image
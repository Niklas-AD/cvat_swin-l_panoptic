import os
import cv2
import numpy as np
import json
import base64
import io
from PIL import Image
import sys
sys.path.insert(0, "Mask2Former")


#install
from skimage.measure import approximate_polygon, find_contours
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config


class ModelHandler:
    def __init__(self, labels):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        print(os.getcwd())
        cfg.merge_from_file("Mask2Former/configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml") #Masterarbeit/Mask2Former/configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml
        cfg.MODEL.WEIGHTS = "model_final_064788.pkl"
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        predictor = DefaultPredictor(cfg)       
        self.model = predictor
        self.labels = labels

    def infer(self, image): 
        results = []
        output = self.model(image)
        result_tensor = output['panoptic_seg'][0].cpu().numpy()
        segments = output['panoptic_seg'][1]
        
        for segment in segments:
            
            mask_instance = ((result_tensor == segment['id']) *255).astype(np.float32)
            contours = find_contours(mask_instance, 0.8)

            for contour in contours:
                contour = np.flip(contour, axis=1)
                
                #delete small polygons
                if contour[:,0].max() - contour[:,0].min() < 30: #50
                    continue
                if contour[:,1].max() - contour[:,1].min() < 50: #90
                    continue 
		
                contour = approximate_polygon(contour, tolerance=2.5)
                if len(contour) < 3:
                    continue

                results.append({
                    "confidence": None,
                    "label": self.labels.get(segment['category_id'], "unknown"),
                    "points": contour.ravel().tolist(),
                    "type": "polygon",
                })

        return results

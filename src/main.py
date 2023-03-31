import supervisely as sly
import warnings

warnings.filterwarnings("ignore")

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict
from dotenv import load_dotenv
import os
from pathlib import Path

app_root_directory = str(Path(__file__).parent.absolute().parents[0])
import sys
sys.path.append(os.path.join(app_root_directory, "scenic"))

import tensorflow as tf
from scenic.projects.owl_vit.configs import clip_b16, clip_b32, clip_l14
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.notebooks import inference
from scenic.model_lib.base_models import box_utils
import numpy as np


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])
model_data_path = os.path.join(root_source_path, "models", "model_data.json")
api = sly.Api()


class OWLViTModel(sly.nn.inference.PromptBasedObjectDetection):
    def get_models(self):
        model_data = sly.json.load_json_file(model_data_path)
        return model_data

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass(self.class_names[0], sly.Rectangle, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        if device.startswith("cuda"):
            # set GPU as visible device
            gpus = tf.config.list_physical_devices("GPU")
            tf.config.set_visible_devices(gpus[0], "GPU")
        else:
            # hide GPUs from visible devices
            tf.config.set_visible_devices([], "GPU")
        # load selected model
        selected_model = self.gui.get_checkpoint_info()["Model"]
        if selected_model == "OWL-ViT base patch 32":
            config = clip_b32.get_config(init_mode="canonical_checkpoint")
        elif selected_model == "OWL-ViT base patch 16":
            config = clip_b16.get_config(init_mode="canonical_checkpoint")
        elif selected_model == "OWL-ViT large patch 14":
            config = clip_l14.get_config(init_mode="canonical_checkpoint")
        module = models.TextZeroShotDetectionModule(
            body_configs=config.model.body,
            normalize=config.model.normalize,
            box_bias=config.model.box_bias,
        )
        variables = module.load_variables(config.init_from.checkpoint_path)
        self.model = inference.Model(config, module, variables)
        self.model.warm_up()
        # define class names
        self.class_names = ["object"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionBBox]:
        # prepare input data
        input_image = sly.image.read(image_path)
        img_height, img_width = input_image.shape[:2]
        if settings["mode"] == "reference_image":
            # get reference image and coordinates of its bbox
            reference_image = api.image.download_np(id=settings["reference_image_id"])
            ref_img_height, ref_img_width = reference_image.shape[:2]
            bbox_coordinates = settings["reference_bbox"]
            class_name = settings["reference_class_name"]
            # add object class to model meta if necessary
            if not self._model_meta.get_obj_class(class_name):
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Rectangle, [255, 0, 0])
                self._model_meta = self._model_meta.add_obj_class(new_class)
            # normalize bounding box coordinates to format required by tensorflow
            bbox_coordinates[0] = bbox_coordinates[0] / ref_img_height
            bbox_coordinates[1] = bbox_coordinates[1] / ref_img_width
            bbox_coordinates[2] = bbox_coordinates[2] / ref_img_height
            bbox_coordinates[3] = bbox_coordinates[3] / ref_img_width
            bbox_coordinates = np.array(bbox_coordinates)
            # pass reference image to model
            reference_embeddings, bbox_idx = self.model.embed_image_query(
                query_image=reference_image,
                query_box_yxyx=bbox_coordinates,
            )
            n_queries = 1  # model does not support multi-query image-conditioned detection
            # get model predictions
            top_query_idx, scores = self.model.get_scores(
                input_image,
                reference_embeddings[None, ...],
                num_queries=1,
            )
            _, _, input_image_boxes = self.model.embed_image(input_image)
            input_image_boxes = box_utils.box_cxcywh_to_yxyx(input_image_boxes, np)
            nms_threshold = settings["nms_threshold"]
            # apply nms to predicted boxes (scores of suppressed boxes will be set to 0)
            for i in np.argsort(-scores):
                if not scores[i]:
                    # this box is already suppressed, continue:
                    continue
                ious = box_utils.box_iou(
                    input_image_boxes[None, [i], :],
                    input_image_boxes[None, :, :],
                    np_backbone=np)[0][0, 0]
                ious[i] = -1.0  # mask self-IoU.
                scores[ious > nms_threshold] = 0.0
            # postprocess model predictions
            confidence_threshold = settings["confidence_threshold"]
            predictions = []
            for box, score in zip(input_image_boxes, scores):
                if score >= confidence_threshold:
                    box[0] = round(box[0] * img_height)
                    box[1] = round(box[1] * img_width)
                    box[2] = round(box[2] * img_height)
                    box[3] = round(box[3] * img_width)
                    score = round(float(score), 2)
                    predictions.append(
                        sly.nn.PredictionBBox(class_name=class_name, bbox_tlbr=box, score=score)
                    )
        elif settings["mode"] == "text_prompt":
            text_queries = settings.get("text_queries")
            text_queries = tuple(text_queries)
            n_queries = len(text_queries)
            if sly.is_production():
                # add object classes to model meta if necessary
                for text_query in text_queries:
                    class_name = text_query.replace(" ", "_")
                    if not self._model_meta.get_obj_class(class_name):
                        self.class_names.append(class_name)
                        new_class = sly.ObjClass(class_name, sly.Rectangle, [255, 0, 0])
                        self._model_meta = self._model_meta.add_obj_class(new_class)
            query_embeddings = self.model.embed_text_queries(text_queries)
            top_query_ind, scores = self.model.get_scores(input_image, query_embeddings, n_queries)
            input_image_features, _, input_image_boxes = self.model.embed_image(input_image)
            input_image_boxes = box_utils.box_cxcywh_to_yxyx(input_image_boxes, np)
            output = self.model._predict_classes_jitted(
                image_features=input_image_features[None, ...],
                query_embeddings=query_embeddings[None, ...],
                )
            labels = np.argmax(output['pred_logits'], axis=-1)
            labels = np.squeeze(labels) # remove unnecessary dimension
            confidence_threshold = settings["confidence_threshold"]
            predictions = []
            for box, label, score in zip(input_image_boxes, labels, scores):
               if score >= confidence_threshold:
                    box[0] = round(box[0] * img_height)
                    box[1] = round(box[1] * img_width)
                    box[2] = round(box[2] * img_height)
                    box[3] = round(box[3] * img_width)
                    label = text_queries[label]
                    label = label.replace(" ", "_")
                    if sly.is_production():
                        class_name = label
                    else:
                        class_name = self.class_names[0]
                    score = round(float(score), 2)
                    predictions.append(
                        sly.nn.PredictionBBox(class_name=class_name, bbox_tlbr=box, score=score)
                    ) 
        return predictions


m = OWLViTModel(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    settings = {}
    settings["mode"] = "text_prompt"
    settings["text_queries"] = ["hummingbird"]
    settings["confidence_threshold"] = 0.1
    results = m.predict(image_path, settings=settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=7)
    print(f"predictions and visualization have been saved: {vis_path}")

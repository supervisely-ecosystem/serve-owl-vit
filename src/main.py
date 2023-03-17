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
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])
model_data_path = os.path.join(root_source_path, "models", "model_data.json")


class OWLViTModel(sly.nn.inference.ObjectDetection):
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
        # load selected model
        selected_model = self.gui.get_checkpoint_info()["Model"]
        if selected_model == "OWL-ViT base patch 32":
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        elif selected_model == "OWL-ViT base patch 16":
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
        elif selected_model == "OWL-ViT large patch 14":
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        self.device = device
        # set model in evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        # define class names
        self.class_names = ["object"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_info(self):
        info = super().get_info()
        info["task type"] = "prompt-based object detection"
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionBBox]:
        # prepare input data
        image = sly.image.read(image_path)
        text_queries = settings.get("text_queries")
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(
            self.device
        )
        if sly.is_production():
            # add object classes to model meta if necessary
            for text_query in text_queries:
                class_name = text_query.replace(" ", "_")
                if not self._model_meta.get_obj_class(class_name):
                    self.class_names.append(class_name)
                    new_class = sly.ObjClass(class_name, sly.Rectangle, [255, 0, 0])
                    self._model_meta = self._model_meta.add_obj_class(new_class)
        # get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.shape[:2]]).to(self.device)
        # convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
        # postprocess model predictions
        predictions = []
        confidence_threshold = settings.get("confidence_threshold", 0.1)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                box = box.cpu().detach().numpy()
                # convert box coordinates from COCO to Supervisely format
                box = [box[1], box[0], box[3], box[2]]
                label = text_queries[label.item()]
                label = label.replace(" ", "_")
                if sly.is_production():
                    class_name = label
                else:
                    class_name = self.class_names[0]
                predictions.append(
                    sly.nn.PredictionBBox(class_name=class_name, bbox_tlbr=box, score=score.item())
                )
        return predictions


m = OWLViTModel(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    settings = {}
    settings["text_queries"] = ["hummingbird"]
    settings["confidence_threshold"] = 0.1
    results = m.predict(image_path, settings=settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=7)
    print(f"predictions and visualization have been saved: {vis_path}")

<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/119248312/229181284-446c6cb0-32e2-47e7-911c-917932f648fd.jpg"/>  

# Serve OWL-ViT

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-owl-vit)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-owl-vit)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-owl-vit.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-owl-vit.png)](https://supervise.ly)

</div>


## Overview

OWL-ViT is an object detection neural network trained on a variety of image - text pairs, which demonstrates outstanding performance on zero-shot text-conditioned and one-shot image-conditioned object detection. It can be used in text prompt mode to detect objects described in text queries or in reference image mode to detect object which was framed by a bounding box on a reference image.

## How To Run

**Pretrained models**

**Step 1.** Select pretrained model architecture and press the **Serve** button:

<img src="https://user-images.githubusercontent.com/91027877/229937093-7af34d05-4fd6-450a-858e-eb0cac8731a8.png" width="80%"/>

**Step 2.** Wait for the model to deploy:

<img src="https://user-images.githubusercontent.com/91027877/229936905-33153fca-5251-4f75-98f2-117f0d3890b3.png" width="80%"/>

**Custom models**

Copy model file path from Team Files and select task type:

https://user-images.githubusercontent.com/91027877/229937368-cd101c7c-57ff-43b5-a4a6-bc8e218c7e18.MP4

# Example: apply OWL-ViT to image in labeling tool

Run NN Image Labeling app, connect to OWL-ViT, write text queries and click on "Apply model to image":

https://user-images.githubusercontent.com/91027877/229938469-be3c3c2a-809f-47a8-ad59-7b91cead3ec4.MP4

If you want to use reference image mode, you can create bounding box for target object and click on apply model to "Apply model to ROI":

https://user-images.githubusercontent.com/115161827/229866610-4e529b09-f2cf-4054-ae1c-acae34ea1fab.mp4

If model predictions look unsatisfying, you can try to tune confidence threshold or nms threshold (if you want to increase number of predicted bounding boxes - decrease confidence threshold and increase nms threshold):

https://user-images.githubusercontent.com/115161827/229866632-9fd8d245-d1a5-454b-a132-3683839000b8.mp4

## Related Apps

You can use deployed model in the following Supervisely Applications ⬇️ 
    
- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>
    
## Acknowledgment

- Based on: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
- Paper: https://arxiv.org/abs/2205.06230

<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/119248312/229181284-446c6cb0-32e2-47e7-911c-917932f648fd.jpg"/>  

# Serve OWL-ViT

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#example-apply-owl-vit-to-image-in-labeling-tool">Example: apply OWL-ViT to image in labeling tool</a> •
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

Copy model file path from Team Files and select model type:

https://user-images.githubusercontent.com/91027877/239012909-2728f5e1-befe-4309-9e2c-3cb22500cfb5.mp4

# Example: apply OWL-ViT to image in labeling tool

Run NN Image Labeling app, connect to OWL-ViT, write text queries and click on "Apply model to image":

https://user-images.githubusercontent.com/91027877/230073611-7934917d-bebe-4825-8a1f-e5343335b0d3.mp4

If you want to use reference image mode, you can create bounding box for target object and click on "Apply model to ROI":

https://user-images.githubusercontent.com/115161827/229866610-4e529b09-f2cf-4054-ae1c-acae34ea1fab.mp4

If model predictions look unsatisfying, you can try to tune confidence threshold or nms threshold (if you want to increase number of predicted bounding boxes - decrease confidence threshold and increase nms threshold):

https://user-images.githubusercontent.com/115161827/229866632-9fd8d245-d1a5-454b-a132-3683839000b8.mp4

## Related Apps

You can use deployed model in the following Supervisely Applications ⬇️ 
    
- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>
    
- [Apply Owl-ViT to Images Project](https://ecosystem.supervise.ly/apps/apply-owl-vit-to-images-project) - app allows to apply OWL-ViT model directly to images project with graphical user interface
    
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-owl-vit-to-images-project" src="https://user-images.githubusercontent.com/115161827/230896644-0ddb3a30-26bf-4468-b1fe-1bfc1d84a3f6.png" height="70px" margin-bottom="20px"/>
    
## Acknowledgment

This app is based on the model `OWL-ViT`. 

- Check it out on [github](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit). ![GitHub Org's stars](https://img.shields.io/github/stars/google-research/scenic?style=social) <br>
- [Paper](https://arxiv.org/abs/2205.06230)

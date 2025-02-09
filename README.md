# Adversarial Pertubation System for License Plate Information Privacy Protection

## Introduction
In order to preserve the privacy of license plate information of public datasets, we pursued the application of Adversarial Attack via Pertubation method.

In our research ([Adversarial Perturbations for License Plate Information Privacy](https://doi.org/10.1007/978-981-96-0434-0_7)), we developed a novel method that is capable of protecting the privacy of data when it is publicly released while ensuring that essential information remains accessible only for authorized operations:

<img src="assets\images\visualize-ideas.png"  width="800">

This repo stores our works in the **Software Architecture** course for the final project, where we has a decent chance to deploy our team's ideas into a multi-modules system. Our system is developed to be flexible, easy-to-maintain as each module can be interchangeable and be handled separately.

## Architecture
Our **Adversarial Pertubation** system consists of following modules:
 - *Models* module: Our system performs adversarial attack on two types of model: detection models and OCR models. In this module, each type of models has an interface that allows us to wrap various detection or OCR models up, in order to easily maintain models within the system, or to add others into the system.

   <img src="assets\images\diagram-models.png"  width="800">

    We currently support the following list of models:
    + YOLOv5 for detection.
    + MTCNN for detection.
    + YOLOv5 for character recognition.
    + EasyOCR for character recognition.

 - *Attacker* module: This module contains functions that are used for the attacking process. The attacking process grants the victim model the authorization to access sensitive information of the dataset. It also contains an *Optimizer* factory, that produce many optimizer functions for the attacking process. We can have different *Attacker* instances with different victim models and different optimizer functions.

    <img src="assets\images\diagram-attacker.png"  width="800">

 - *InferAttack* module: This is the module that defines the de-identify methods and directly perform the attacking process on the post-de-identified dataset using specified *Attacker* instances.

   <img src="assets\images\diagram-inferattack.png"  width="800">

 - *Evaluater* module: This module performs the evaluation of attacked models on postprocessed dataset. We can visualize how attacked models works before and after the attacking processes.

   <img src="assets\images\diagram-evaluater.png"  width="800">

   The module also has some metrics calculation in order to provide in-depth analysis on how the attacking process enable its victims to access the de-identified information, such as:
   + IoU (Intersection over Union) score.
   + CER (Characters error rate) score.
   + Similarity score.

## Demo
Our demo for the project: [Demo video](https://youtu.be/dW7NMLYdZA0).
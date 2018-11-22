<img src='https://github.com/gqfiddler/sketch2pic/blob/master/cover_image.png?raw=true'>

# Overview
Product design is a multidisciplinary role that incorporates artistic vision, functional awareness, psychological insight, and market analysis.  For physical product design (e.g. shoes, apparel, or cars), the ability to envision a product in high detail and analyze its appeal is a crucial skill.  Once a compelling vision has been developed, a designer must also find a way, through words or painstaking artistic renderings, to communicate this vision to others in the company.

In their 2016 paper <a href='https://arxiv.org/abs/1611.07004'>“Image-to-Image Translation with Conditional Adversarial Networks,”</a> commonly known as the "pix2pix" paper, Philip Isola et al. present a methodology and architecture for generating a full-fledged photorealistic image based on some substrata or idealization of that image (e.g. a B&W image, map, or outline) using Conditional Generative Adversarial Networks (cGANs).  Where GANs typically take random noise as an input, cGANs take some seed image that is paired to a target image: in the designer's case, these could be a quick sketch and a full-fledged photograph.

In this project, I use the architecture Isola et al created to develop a tool for designers that I am calling sketch2pic.  This tool takes a basic hand-drawn sketch of their product design and produces photo-like renderings of how that product might look once fully realized.  While this tool is still fairly low-resolution due largely to dataset and compute constrictions, a moderately upscaled version could enhance the speed and specificity with which designers can brainstorm ideas – and it also has the potential to communicate their ideas quickly and compellingly to supervisors, colleagues, and clients.

# Additions and innovations
This project includes several differences from the original implementation of the original pix2pix algorithms.  Some of these are highly technical and will be addressed in the notebook (several of these tweaks allow for slightly faster and more precise training).  Two large-scale, practical aspects are worth noting here, however:
- This tool gives you the ability to generate your own dataset tailored to your preferred level of detail.  I've built a basic and incredibly easy-to-use image preprocessing function that can generate sketch pairings for photographs.  This includes three easy, simple-to-tweak parameters that allow you to specify how detailed of images you want the model to prepare for, depending on whether the designer is aiming for a first-draft idea model, or a more detailed fleshing out of a later draft.
- This tool allows you to generate multiple ideas for each sketch.  There are several layers here: first, the model has random dropout layers that create slight differences with every run.  Second, I've included both the traditional pix2pix version and my own tweaked algorithm, which can generate slightly different results.  Third, there's a nearly-completed (still under construction) algorithm here ('cGAN_multi_out.py') that implements a version trained to produce several quite different photo-like images for each sketch.
- This tool is extremely easy to use.  While the backend architecture is quite complicated, the tool consists of two python files.  Using the tool in a notebook takes about five lines of code.  Significant compute is required for the initial training of the model, but that is the only practical constraint to easy use.

# Dataset
For the purposes of this demonstration, I've used the readily available <a href='http://vision.cs.utexas.edu/projects/finegrained/utzap50k/'>Zappos 50K dataset</a> from the University of Texas, which consists of 50,000 images of shoes.  This dataset was used as one of the example datasets in the original paper.  However, it's important to note that, since this tool includes dataprocessing, the model can be used on any dataset of images of a product type against a blank background.

# Files
- The main presentation file here, for an overview and explanation of the project, is the Final Capstone - Sketch2pic notebook
- the image loading and processing functions are in data_loading.py
- the main model is in cGAN_functions.py
- you can upload pre-trained weights to that model from full_cGAN_weights.h5 and full_disc_weights.h5
- a version of the model with more traditional pix2pix training (without using the set_weight() gradient trick described in the notebook) is in cGAN_functions_original.py
- a multi-output version that generates multiple images for each sketch is at cGAN_multi_out.py; however, it still has a few bugs to be worked out.


# Use and exploration
While this example is fairly specific, pix2pix-based models are broadly generalizable to image-to-image tasks.  This could be used for lots of different purposes, and there are many tweaks and alternate versions still to be explored.  Feel free to download this version and play around with it - and let me know if you make any interesting discoveries!

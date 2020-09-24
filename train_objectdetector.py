import os
from comet_ml import Experiment
from deepforest import deepforest
from deepforest import preprocess
from deepforest import utilities
from deepforest import __version__


trained_model = deepforest.deepforest(saved_model='palm_detection_model.h5')

# Configurations
trained_model.config["epochs"] = 200
trained_model.config["batch_size"] = 1
trained_model.config["backbone"] = 'resnet101'
trained_model.config["multi-gpu"] = 0
trained_model.config["save-snapshot"] = True
trained_model.config["snapshot_path"] = "/checkpoint"
trained_model.config["random_transform"] = True

#Input
image_dir = input("Input directory samples image:  ")
annotations_file = os.path.join(image_dir, "annotations.csv")

#Training Process
trained_model.train(annotations=annotations_file, input_type="fit_generator")
#Save Model
trained_model.model.save("/palm_model_retinanet101_recent.h5")
#Evaluate Model
trained_model.evaluate_generator(annotations=annotations_file)



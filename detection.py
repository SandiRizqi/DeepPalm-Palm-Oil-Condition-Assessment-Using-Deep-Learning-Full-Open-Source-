import numpy as np
from PIL import Image
import os
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from deepforest import deepforest
import geopandas
import rasterio
import descartes
import shapely
from classification import classification
from config import predictor

#convert hand annotations from shp into DeepForest format
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 10**1000000
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


inputs = input("Masukkan Alamat Directory: ")
extension = input("Masukkan Extension Format File: ")
jenis_detector = input("TM atau TM&TBM: ")
classify = input("Apakah Termasuk Klasifikasi Object?: ")

#Input Models
if jenis_detector == "TM":
    model = predictor('palm_detection_model.h5')
else:
    model = predictor('palm_tm_tbm_detection_model.h5')


classifier = tensorflow.keras.models.load_model('InceptionV3_pokok_kuning_classifier_model.h5')

#Inferencing Process
lists = os.listdir(inputs)
files = [i for i in lists if i.endswith(extension)]
print("Daftar Deteksi:")
for index in range(len(files)):
    print(files[index])
    
for index in range(len(files)):
    print("Proses Deteksi", files[index])
    rasterpath = inputs + files[index]
    prediction = model.predict_tile(rasterpath, patch_size=1000,patch_overlap=0.2, return_plot=False,iou_threshold=0.18)
    #convert rowcol to geograph
    with rasterio.open(rasterpath) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY  = dataset.res
    prediction["xmin"] = (prediction["xmin"] * pixelSizeX) + bounds.left
    prediction["xmax"] = (prediction["xmax"] * pixelSizeX) + bounds.left
    prediction["ymin"] = bounds.top - (prediction["ymin"] * pixelSizeY) 
    prediction["ymax"] = bounds.top - (prediction["ymax"] * pixelSizeY)
    
    prediction['geometry'] = prediction.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    prediction = geopandas.GeoDataFrame(prediction, geometry='geometry')
    prediction.crs = {'init' :'epsg:4326'}
    
    if classify == "True":
        classification(rasterpath, classifier, prediction)
    else:
        pass
    prediction.to_file(inputs + files[index][:-4] + ".shp", driver='ESRI Shapefile',crs=prediction.crs)
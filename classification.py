import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
import rasterio
import geopandas
from PIL import Image


#Input Classifier
def classification(rasterpath, classifier, prediction):
    #backto row col index
    with rasterio.open(rasterpath) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY  = dataset.res
    
    #backto row col index
    prediction["xmin"] = (prediction["xmin"].astype('float32') - bounds.left)  / pixelSizeX
    prediction["xmax"] = (prediction["xmax"].astype('float32') - bounds.left) / pixelSizeX
    prediction["ymin"] = (bounds.top - prediction["ymin"].astype('float32')) / pixelSizeY
    prediction["ymax"] = (bounds.top - prediction["ymax"].astype('float32')) / pixelSizeY
    prediction["category"] = ""
    
    #Object Classification
    image = Image.open(rasterpath)
    numpy_image = np.array(image)
    for i in range(prediction.index.size):
        chip = numpy_image[prediction['ymin'].loc[i].astype('int'):prediction['ymax'].loc[i].astype('int'),prediction['xmin'].loc[i].astype('int'):prediction['xmax'].loc[i].astype('int'),:]
        img = Image.fromarray(chip)
        img = img.resize((224,224))
        test = img_to_array(img)
        test = test / 255
        test = np.expand_dims(test, axis=0)
        result = classifier.predict(test)
        prediction["category"].loc[i] = float(result)
        print('{}/{}'.format(i+1, prediction.index.size), result)
    
    return prediction
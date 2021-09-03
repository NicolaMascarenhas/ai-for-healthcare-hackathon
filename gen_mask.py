import tensorflow as tf
import numpy as np
from PIL import Image
import io

def show(model, img):
    h, w = img.size
    img_resized = img.resize((256, 256))
    img = tf.keras.preprocessing.image.img_to_array(img_resized)
    img = np.expand_dims(img, axis = 0)
    img = img / 255
    pred= model.predict(img)

    pred1 = np.argmax(pred, axis=-1)
    pred1 = np.squeeze(pred1)

    pred_arg = pred1
    rescaled = (255.0 / pred_arg.max() * (pred_arg - pred_arg.min())).astype(np.uint8)
    mask_img = Image.fromarray(rescaled)
    
    mask_img = mask_img.resize((h, w))

    return mask_img
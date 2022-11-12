from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
model = load_model("fruit.h5")
img = image.load_img('applehealthy.JPG',target_size=(128,128))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=model.predict(x)
classes=np.argmax(pred,axis=1)
print(classes)

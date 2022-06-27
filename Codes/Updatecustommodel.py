import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials
from numpy import source

firebase_admin.initialize_app(
    credentials.Certificate('newprivatekey.json'),
    options={
        'storageBucket': 'bucketname.appspot.com',
    }
)

model= ml.get_model("18859661")
source=ml.TFLiteGCSModelSource.from_tflite_model_file('Unmask_the_masked.tflite')
model.model_format=ml.TFLiteFormat(model_source=source)

model.display_name="Unmasking_DL_model"

updated_model= ml.update_model(model)
ml.publish_model(updated_model.model_id)
import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials

firebase_admin.initialize_app(
    credentials.Certificate('newprivatekey.json'),
    options={
        'storageBucket': 'bucketname.appspot.com',
    }
)

source  =ml.TFLiteGCSModelSource.from_tflite_model_file('Unmask_the_masked.tflite')
tflite_format= ml.TFLiteFormat(model_source=source)

model =ml.Model(
    display_name="Unmasking_DL_model",
    model_format=tflite_format
)

new_model = ml.create_model(model)
ml.publish_model(new_model.model_id)
print(new_model.model_id)
print(new_model.display_name)
print(new_model.model_format)
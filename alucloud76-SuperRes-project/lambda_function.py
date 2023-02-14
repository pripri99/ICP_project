import json
import boto3
import datetime
from PIL import Image
import io
# import matplotlib.image as mpimg
import numpy as np
from datetime import date

# import matplotlib.image as mpimg
# from generator import build_generator_model
from keras.models import load_model
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflowjs as tfjs
# import tflite_runtime.interpreter as tflite

s3 = boto3.client("s3")

def tf_lite_test():
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./generator.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    #input_image = imageio.imread("./0806.jpg", pilmode="RGB").astype(float)
    #input_image_lr = resize(input_image, input_shape[1:])
    image = Image.open("./0806.jpg")
    new_image = image.resize(input_shape[1:3])
    rgb_im = new_image.convert('RGB')
    input_image_lr = np.array(rgb_im).astype(float)

    input_data = np.array([input_image_lr], dtype=np.float32) / 127.5 - 1.0

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("output data:")
    print(output_data.shape, type(output_data))
    output_data =0.5 * output_data + 0.5
    print(output_data.shape)
    plt.imshow(output_data[0])
    plt.axis('off')
    plt.savefig("./test_output_tflite.jpg")
    plt.close()



def main():
    # Convert the model
    generator = load_model("./generator_srgan")
    # input_image = imageio.imread("./0806.jpg", pilmode="RGB").astype(float)
    # input_image_lr = resize_low_res_image(input_image, low_res_dimensions=(64, 64))
    # input_image_lr = resize(input_image, (64, 64))
    image = Image.open("./0806.jpg")
    new_image = image.resize((64,64))
    rgb_im = new_image.convert('RGB')
    input_image_lr = np.array(rgb_im).astype(float)

    imgs_lr = np.array([input_image_lr]) / 127.5 - 1.0

    output_image_hr = generator.predict(imgs_lr)
    print(output_image_hr.shape)

    output_image_hr =0.5 * output_image_hr + 0.5
    print(output_image_hr.shape)
    plt.imshow(output_image_hr[0])
    plt.axis('off')
    plt.savefig("./test_output.jpg")
    plt.close()

    # convert to tensorflowjs
    tfjs.converters.save_keras_model(generator, "./generator_tfjs")


    # converter = tf.lite.TFLiteConverter.from_keras_model("./generator_srgan")
    # tflite_model = converter.convert()

    # Save the model.
    # with open('./generator.tflite', 'wb') as f:
        # f.write(tflite_model)


def load_image_from_s3(local_image_file_name, image_object_name, input_bucket_name, low_res_dimensions=(64, 64)):
    with open(local_image_file_name, "wb") as f:
        s3.download_fileobj(input_bucket_name, image_object_name, f)
        # img = mpimg.imread(local_image_file_name)
        # img = Image.open(local_image_file_name)
        # input_img = np.asarray(img)
        # input_img.astype(np.float)
        '''img = mpimg.imread(local_image_file_name)
        input_img = np.asarray(img)
        input_img.astype(np.float)'''
        # input_img = imageio.imread(local_image_file_name, pilmode="RGB").astype(np.float)
        image = Image.open(local_image_file_name)
        new_image = image.resize(low_res_dimensions)
        rgb_im = new_image.convert('RGB')
        input_img = np.array(rgb_im).astype(np.float)
        


    return input_img


def resize_low_res_image(input_img, low_res_dimensions=(64, 64)):
    img_lr = resize(input_img, low_res_dimensions)
    return img_lr


def save_image_to_s3(img_numpy, image_id, target_bucket_name):
    img_bytes = io.BytesIO()
    img_numpy.save(img_bytes, format="JPG")
    # fig.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)
    day = str(date.today())
    now = datetime.datetime.now()
    path = "output/" + day + "/" + str(now.time()) + "/plan_" + str(image_id) + ".png"
    s3.upload_fileobj(img_bytes, target_bucket_name, path)
    """resource = boto3.resource("s3")
    object = resource.Object(target_bucket_name, path)
    object.copy_from(
        CopySource={"Bucket": target_bucket_name, "Key": path},
        MetadataDirective="REPLACE",
        ContentType="image/png",
    )"""
    return path


def lambda_handler(event, context):
    print("event is:", event)
    # load image from input s3 bucket
    # increase resolution of image
    # save high res image in output s3 bucket
    # generator = build_generator_model()
    # generator = load_model("./generator_srgan")
    # generator = tf.saved_model.load("./generator_srgan")

    input_image_lr = load_image_from_s3(
        "./tmp_input", image_object_name, input_bucket_name="alucloud76-low-res-input"
    )
    #input_image_lr = resize_low_res_image(input_image, low_res_dimensions=(64, 64))

    #output_image_hr = generator.predict(input_image_lr)

    img_output_path = save_image_to_s3(
        output_image_hr, image_id, "alucloud76-high-res-output"
    )

    return {
        "statusCode": 200,
        "body": json.dumps("Hello from Lambda!"),
        "generated": img_output_path,
    }

if __name__ == "__main__":
    main()
    tf_lite_test()

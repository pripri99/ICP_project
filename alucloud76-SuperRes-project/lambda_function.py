import json
import boto3
import datetime
from PIL import Image
import io
import numpy as np
import io
import random

from oil_painting import oilpaint


s3 = boto3.client("s3")


def load_image_from_s3(
    image_object_name, input_bucket_name, low_res_dimensions=(64, 64)
):
    file_stream = io.BytesIO()
    s3.download_fileobj(input_bucket_name, image_object_name, file_stream)
    image = Image.open(file_stream)
    if low_res_dimensions is not None:
        image = image.resize(low_res_dimensions)
    rgb_im = image.convert("RGB")
    input_img = np.array(rgb_im)

    return input_img


def save_image_to_s3(img_numpy, image_id, target_bucket_name):
    img_final = Image.fromarray(img_numpy)
    img_bytes = io.BytesIO()
    img_final.save(img_bytes, "png")
    img_bytes.seek(0)
    now = datetime.datetime.now()
    path = image_id.split(".")[0] + "_" + str(now.time()) + ".png"
    s3.upload_fileobj(img_bytes, target_bucket_name, path)
    return path


def lambda_handler(event, context):
    # print("event is:", event)
    data = event["Records"][0]
    print("Keys:", [k for k in data])

    input_image_lr = load_image_from_s3(
        image_object_name=data["s3"]["object"]["key"],
        input_bucket_name=data["s3"]["bucket"]["name"],
        low_res_dimensions=None,
    )

    print("input image loaded")

    # output_image_hr = generator.predict(input_image_lr)
    print("type:", type(input_image_lr))
    print("shape:", input_image_lr.shape)
    output_image_hr = input_image_lr

    img_output_path = save_image_to_s3(
        img_numpy=output_image_hr,
        image_id=data["s3"]["object"]["key"],
        target_bucket_name="alucloud76-high-res-output",
    )

    return {
        "statusCode": 200,
        "body": json.dumps("Hello from Lambda!"),
        "generated": img_output_path,
    }


def tf_lite_test():
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./generator.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]["shape"]
    image = Image.open("./0806.jpg")
    new_image = image.resize(input_shape[1:3])
    rgb_im = new_image.convert("RGB")
    input_image_lr = np.array(rgb_im).astype(float)

    input_data = np.array([input_image_lr], dtype=np.float32) / 127.5 - 1.0

    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print("output data:")
    print(output_data.shape, type(output_data))
    output_data = 0.5 * output_data + 0.5
    print(output_data.shape)
    plt.imshow(output_data[0])
    plt.axis("off")
    plt.savefig("./test_output_tflite.jpg")
    plt.close()


def main():
    # Convert the model
    generator = load_model("./generator_srgan")
    image = Image.open("./0806.jpg")
    new_image = image.resize((64, 64))
    rgb_im = new_image.convert("RGB")
    input_image_lr = np.array(rgb_im).astype(float)

    imgs_lr = np.array([input_image_lr]) / 127.5 - 1.0

    output_image_hr = generator.predict(imgs_lr)
    print(output_image_hr.shape)

    output_image_hr = 0.5 * output_image_hr + 0.5
    print(output_image_hr.shape)
    plt.imshow(output_image_hr[0])
    plt.axis("off")
    plt.savefig("./test_output.jpg")
    plt.close()

    # convert to tensorflowjs
    tfjs.converters.save_keras_model(generator, "./generator_tfjs")

    # converter = tf.lite.TFLiteConverter.from_keras_model("./generator_srgan")
    # tflite_model = converter.convert()

    # Save the model.
    # with open('./generator.tflite', 'wb') as f:
    # f.write(tflite_model)


if __name__ == "__main__":
    # main()
    # tf_lite_test()
    image = Image.open("./0806.jpg")
    rgb_im = image.convert("RGB")
    input_img = np.array(rgb_im)

    rdn_num = random.randint(0,800)
    output_img = oilpaint(input_img, brush_radius=5, effect_intensity=1)
    print(output_img.shape)
    img_final = Image.fromarray(output_img)
    img_final.save("./painted_img"+str(rdn_num)+".png","PNG")


import json
import boto3
import datetime
import io
from datetime import date
s3 = boto3.client("s3")

def save_image_to_s3(img_numpy, image_id):
    img_bytes = io.BytesIO()
    img_numpy.save(img_bytes, format="JPG")
    # fig.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)
    day = str(date.today())
    now = datetime.datetime.now()
    path = (
        "output/"
        + day
        + "/"
        + str(now.time())
        + "/plan_"
        + str(image_id)
        + ".png"
    )
    s3.upload_fileobj(img_bytes, "plan-output-temp", path)
    resource = boto3.resource("s3")
    object = resource.Object("plan-output-temp", path)
    object.copy_from(
        CopySource={"Bucket": "plan-output-temp", "Key": path},
        MetadataDirective="REPLACE",
        ContentType="image/png",
    )
    return path

def lambda_handler(event, context):
    # load image from input s3 bucket
    # increase resolution of image
    # save high res image in output s3 bucket
    
    return {"statusCode": 200, "body": json.dumps("Hello from Lambda!")}

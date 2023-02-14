//const tf = require("@tensorflow/tfjs");

// const tfn = require("@tensorflow/tfjs-node");
import * as tf from "@tensorflow/tfjs-node";
import { Tensor3D } from "@tensorflow/tfjs-node";

const fs = require("fs");
const imageGet = require("get-image-data");

const Jimp = require("jimp");

// Directory path for model files (model.json, metadata.json, weights.bin)
// NOTE: It can be obtained from [Export Model] -> [Tensorflow.js] -> [Download my model]
//        on https://teachablemachine.withgoogle.com/train/image
const MODEL_DIR_PATH = `generator_tfjs`;

// Path for image file to predict class
const IMAGE_FILE_PATH = `0806.jpg`;

const INPUT_IMAGE_HEIGHT = 64;
const INPUT_IMAGE_WIDTH = 64;
const INPUT_IMAGE_CHANNELS = 3;

const OUTPUT_IMAGE_HEIGHT = 256;
const OUTPUT_IMAGE_WIDTH = 256;
const OUTPUT_IMAGE_CHANNELS = 3;

const handler = tf.io.fileSystem("./generator_tfjs/model.json");

async function main() {
  const model = await tf.loadGraphModel(handler);
  var s3 = new AWS.S3({ apiVersion: "2006-03-01" });
  var params = { Bucket: "myBucket", Key: "myImageFile.jpg" };
  var file = require("fs").createWriteStream("/path/to/file.jpg");
  s3.getObject(params).createReadStream().pipe(file);

  /*let img = fs.readFileSync(process.argv[2] || "0806.jpg");
  const im = tf.node.decodeJpeg(img).toFloat().expandDims();
  const res = model.predict(im);

  let outputTensor = tf.squeeze(res, [0]);

  const outputImage = await tf.node.encodeJpeg(outputTensor);
  fs.writeFileSync("esrgan.jpeg", outputImage);*/
}

main();

exports.handler = async (event) => {
  bucketName = "<your-bucket-name>";
  const MODEL_URL = `https://${bucketName}.s3.amazonaws.com/model.json`;
  const model = await tf.loadLayersModel(MODEL_URL);
  let x = parseFloat(event.x);
  input = tf.tensor([x]);
  const result = model.predict(input);
  const y = (await result.array())[0][0];
  return {
    statusCode: 200,
    body: JSON.stringify(y),
  };
};

/*console.log('Loading function');

const aws = require('aws-sdk');

const s3 = new aws.S3({ apiVersion: '2006-03-01' });


exports.handler = async (event, context) => {
    //console.log('Received event:', JSON.stringify(event, null, 2));

    // Get the object from the event and show its content type
    const bucket = event.Records[0].s3.bucket.name;
    const key = decodeURIComponent(event.Records[0].s3.object.key.replace(/\+/g, ' '));
    const params = {
        Bucket: bucket,
        Key: key,
    };
    try {
        const { ContentType } = await s3.getObject(params).promise();
        console.log('CONTENT TYPE:', ContentType);
        return ContentType;
    } catch (err) {
        console.log(err);
        const message = `Error getting object ${key} from bucket ${bucket}. Make sure they exist and your bucket is in the same region as this function.`;
        console.log(message);
        throw new Error(message);
    }

        const response = {
        statusCode: 200,
        body: JSON.stringify('Hello from Lambda!'),
    };
    return response;

};
*/

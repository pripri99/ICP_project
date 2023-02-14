//const tf = require("@tensorflow/tfjs");

const tfn = require("@tensorflow/tfjs-node");
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

function improve_resolution(inputTensor) {
  return tf.tidy(() => {
    const res = decoder.predict(inputTensor).mul(255).cast("int32");
    const reshaped = res.reshape([
      inputTensor.shape[0],
      OUTPUT_IMAGE_HEIGHT,
      OUTPUT_IMAGE_WIDTH,
      OUTPUT_IMAGE_CHANNELS,
    ]);
    return reshaped;
  });
}

(async () => {
  //const labels = require(`${MODEL_DIR_PATH}/metadata.json`).labels;

  const model = await tf.loadLayersModel(`file://${MODEL_DIR_PATH}/model.json`);
  model.summary();

  const image = await Jimp.read(IMAGE_FILE_PATH);
  image.cover(
    INPUT_IMAGE_HEIGHT,
    INPUT_IMAGE_WIDTH,
    Jimp.HORIZONTAL_ALIGN_CENTER | Jimp.VERTICAL_ALIGN_MIDDLE
  );

  const NUM_OF_CHANNELS = 3;
  let values = new Float32Array(64 * 64 * NUM_OF_CHANNELS);

  let i = 0;
  image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
    const pixel = Jimp.intToRGBA(image.getPixelColor(x, y));
    pixel.r = pixel.r / 127.0 - 1;
    pixel.g = pixel.g / 127.0 - 1;
    pixel.b = pixel.b / 127.0 - 1;
    pixel.a = pixel.a / 127.0 - 1;
    values[i * NUM_OF_CHANNELS + 0] = pixel.r;
    values[i * NUM_OF_CHANNELS + 1] = pixel.g;
    values[i * NUM_OF_CHANNELS + 2] = pixel.b;
    i++;
  });

  const outShape = [64, 64, NUM_OF_CHANNELS];
  let img_tensor = tf.tensor3d(values, outShape, "float32");
  img_tensor = img_tensor.expandDims(0);

  const predictions = await model.predict(img_tensor).dataSync();
  console.log(predictions);

  /*for (let i = 0; i < predictions.length; i++) {
    const label = labels[i];
    const probability = predictions[i];
    console.log(`${label}: ${probability}`);
  }*/
})();

/*exports.handler = async (event) => {
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
};*/

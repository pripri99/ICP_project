console.log("Loading function");
// Import required AWS SDK clients and commands for Node.js.
//import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
const { S3Client, GetObjectCommand } = require("@aws-sdk/client-s3");
const tf = require("@tensorflow/tfjs");

// import { s3Client } from "./libs/s3Client.js"; // Helper function that creates an Amazon S3 service client module.

const fs = require("fs");

var path = "image.jpg";

exports.handler = async (event, context) => {
  //console.log('Received event:', JSON.stringify(event, null, 2));
  const s3Client = new S3Client({ region: "us-east-1" });

  // Get the object from the event and show its content type
  const bucket = event.Records[0].s3.bucket.name;
  const key = decodeURIComponent(
    event.Records[0].s3.object.key.replace(/\+/g, " ")
  );
  const bucketParams = {
    Bucket: bucket,
    Key: key,
  };
  try {
    const data = await s3Client.send(new GetObjectCommand(bucketParams));
    await data.Body.pipe(createWriteStream(path));
    const buf = fs.readFileSync(path);
    const img = tf.node.decodeJpeg(buf);
    const dataString = await data.Body.transformToString();
    console.log("dataString:", dataString);
    const response = {
      statusCode: 200,
      body: JSON.stringify("Hello from Lambda!"),
      result: JSON.stringify(dataString),
    };

    //return ContentType;
    return response;
  } catch (err) {
    console.log("Error", err);
    const message = `Error getting object ${key} from bucket ${bucket}. Make sure they exist and your bucket is in the same region as this function.`;
    console.log(message);
    throw new Error(message);
  }
};

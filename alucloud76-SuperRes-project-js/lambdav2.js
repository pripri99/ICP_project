console.log("Loading function");
// Import required AWS SDK clients and commands for Node.js.
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { s3Client } from "./libs/s3Client.js"; // Helper function that creates an Amazon S3 service client module.

exports.handler = async (event, context) => {
  //console.log('Received event:', JSON.stringify(event, null, 2));

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

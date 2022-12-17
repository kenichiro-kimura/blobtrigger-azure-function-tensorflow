import logging
import azure.functions as func
import io
import tensorflow as tf
from PIL import Image
import numpy as np
from azure.storage.blob import BlobServiceClient,BlobClient
import os
import requests

URL = "https://notify-api.line.me/api/notify";

def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")

    graph_def = tf.compat.v1.GraphDef()
    labels = []

    filename = "model.pb"
    labels_filename = "labels.txt"

    # Create blobclient to download model
    blob_service_client = BlobServiceClient.from_connection_string(os.environ.get('ConnectionStrings:StrageConnectionString'))
    container_name = myblob.name.split('/')[0]
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    
    # Import the TF graph   
    download_stream = blob_client.download_blob();
    graph_def.ParseFromString(download_stream.readall())
    tf.import_graph_def(graph_def, name='')

    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    # Load a image from blob
    image_buffer = io.BytesIO(myblob.read())
    image = Image.open(image_buffer)

    # Crop and resize
    image = image.crop((650,378,650 + 128, 378 + 128)).resize((224,224))

    # Convert to OpenCV format
    image = convert_to_opencv(image)

    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions = sess.run(prob_tensor, {input_node: [image] })
        except KeyError:
            logging.error("Couldn't find classification output layer: " + output_layer + ".")
            logging.error("Verify this a model exported from an Object Detection project.")
            return

    # Print the highest probability label
    highest_probability_index = np.argmax(predictions)
    logging.info('Classified as: ' + labels[highest_probability_index] + '(' + str(predictions[0][highest_probability_index]) + ')')

    # Or you can print out all of the results mapping labels to probabilities.
    label_index = 0
    for p in predictions:
        truncated_probablity = np.float64(np.round(p,8))
        logging.info(labels[label_index] + "," + str(truncated_probablity))
        label_index += 1    

    image_buffer.seek(0)
    if(labels[highest_probability_index] == "open"):
        send_linenotify(str(predictions[0][highest_probability_index]),image_buffer.getvalue())

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image
 
def send_linenotify(probability,image):
    line_token = os.environ.get('LineToken')
    headers = {"Authorization": "Bearer %s" % line_token};

    message = "玄関の鍵が開いてるかもしれません！\n(確率 %s)\n" % probability
    data = {
        "message": message.encode("utf-8"),
    }

    files = {
        "imageFile": image
    }

    requests.post(URL, headers=headers, data=data, files=files)    
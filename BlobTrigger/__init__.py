import logging
import azure.functions as func
import io
import tensorflow as tf
from PIL import Image
import numpy as np
#import cv2
import ctypes
import importlib
from datetime import datetime, timedelta
from azure.storage.blob import generate_blob_sas,BlobSasPermissions,BlobServiceClient,BlobClient
import os
import requests

URL = "https://notify-api.line.me/api/notify";

# cv2を使うためにfunctionsのコンテナにないライブラリを無理矢理読み込む
exlibpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/tmp/'
ctypes.CDLL(exlibpath + 'libglib-2.0.so.0')
ctypes.CDLL(exlibpath + 'libgthread-2.0.so.0')

cv2 = importlib.import_module('cv2')

def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")

    blob_service_client = BlobServiceClient.from_connection_string(os.environ.get('CUSTOMCONNSTR_StrageConnectionString'))
    blob_names = myblob.name.split('/')
    container_name = blob_names.pop(0)
    blob_name = "/".join(blob_names)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    sas = get_blob_sas(blob_service_client,container_name,blob_name);
    blob_url = BlobClient.from_blob_url(blob_client.url,credential=sas).url

    graph_def = tf.compat.v1.GraphDef()
    labels = []

    # These are set to the default names from exported models, update as needed.
    filename = "model.pb"
    labels_filename = "labels.txt"

    # Import the TF graph
    pb_blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    
    download_stream = pb_blob_client.download_blob();
    graph_def.ParseFromString(download_stream.readall())
    tf.import_graph_def(graph_def, name='')

    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    # Load from blob
    image = Image.open(io.BytesIO(myblob.read()))    
    image = image.crop((650,378,650 + 128, 378 + 128))

    # Convert to OpenCV format
    image = convert_to_opencv(image)

    # Resize that square up to 224x224
    image = cv2.resize(image, (224,224), interpolation = cv2.INTER_LINEAR)

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

    if(labels[highest_probability_index] == "open"):
        send_linenotify(str(predictions[0][highest_probability_index]),blob_url])
    # Or you can print out all of the results mapping labels to probabilities.
    label_index = 0
    for p in predictions:
        truncated_probablity = np.float64(np.round(p,8))
        logging.info(labels[label_index] + "," + str(truncated_probablity))
        label_index += 1    

    logging.info(blob_url)

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image
 
def get_blob_sas(blob_client,container_name, blob_name):
    sas_blob = generate_blob_sas(account_name=blob_client.account_name, 
                                container_name=container_name,
                                blob_name=blob_name,
                                account_key=blob_client.credential.account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(hours=1))
    return sas_blob

def send_linenotify(probability,image_url):
    line_token = os.environ.get('LineToken')
    headers = {"Content-Type": "application/x-www-form-urlencoded","Authorization": "Bearer %s" % line_token};

    message = "玄関の鍵が開いてるかもしれません！\n(確率 %s)\n%s\n" % probability,image_url
    data = {"message": message.encode("utf-8")}
    requests.post(URL, headers=headers, data=data);
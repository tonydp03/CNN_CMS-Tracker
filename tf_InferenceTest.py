import argparse 
import tensorflow as tf
import pandas as pd
import numpy as np
import random

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="pixel_only_final.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
        #print(op.name)
        
    # We access the input and output nodes

    x = graph.get_tensor_by_name('prefix/hit_shape_input:0')
    y = graph.get_tensor_by_name('prefix/output/Softmax:0')
    #print x
    #print('\n\n')
    #print y

    hdfFile = pd.read_hdf("pixel_only_data_test.h5")
    #print hdfFile

    nrows = hdfFile.shape[0]
    #print 'Number of images: ', nrows
    print '\nSelecting one image randomly...'
    testIdx = random.randint(0, nrows)
    print '\nSelected image n', testIdx    
    testRow = hdfFile.loc[testIdx].values
    inputNumber = len(testRow)
    #print inputNumber
    #print '\n'
    #print testRow

    inputData = testRow[:(inputNumber-2)] 
    labelOutput = testRow[inputNumber-2:]

    #print inputData
    #print '\n'
    #print labelOutput

    inputData = np.reshape(inputData, (1,20,16,16))
    #print inputData

    # We launch a Session
    with tf.Session(graph=graph) as sess:

        y_out = sess.run(y, feed_dict={x: inputData})


        print 'Expected output: ', labelOutput
        print '\n'
        print 'Network output: ', y_out


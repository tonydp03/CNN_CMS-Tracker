/* A simple C++ program to import an UFF model, read input data from a HDF file and make fast inference on GPU with NVIDIA TensorRT */


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <ctime>

#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

#include "H5Cpp.h"
using namespace H5;

using namespace nvuffparser;
using namespace nvinfer1;
#include "common.h"


#define MAX_WORKSPACE (1 << 20)
#define maxBatchSize 60000

//define the data info
const H5std_string FILE_NAME("pixel_only_data_test.h5");
const H5std_string DATASET_NAME_DATA("data/block0_values");
const H5std_string DATASET_NAME_LABELS("labels/block0_values");

// Attributes of the model
static const int INPUT_CH = 20;
static const int INPUT_H = 16;
static const int INPUT_W = 16;
static const int OUTPUT_SIZE = 2;
const char* INPUT_TENSOR_NAME = "hit_shape_input";
const char* OUTPUT_TENSOR_NAME = "output/Softmax";
const std::string dir{"./"};
const std::string fileName{dir + "pixel_only_final.uff"};


int main(int argc, char** argv){
    
  std::cout << "*** MODEL TO IMPORT: " << fileName << "\n";
  std::cout << "*** DATASET FILE: " << FILE_NAME << "\n";
  std::cout << "*** MAX WORKSPACE: " << MAX_WORKSPACE << "\n";
  std::cout << "*** MAX BATCHSIZE: " << maxBatchSize << std::endl;

  int batchSize = 30000;
  std::cout << "*** Number of images to process (batchSize): " << batchSize << std::endl;

  float ms;
  Logger gLogger; // object for warning and error reports

  // *** IMPORTING THE MODEL *** 
  std::cout << "*** IMPORTING THE UFF MODEL ***" << std::endl;

  // Create the builder and the network
  IBuilder* builder = createInferBuilder(gLogger);
  INetworkDefinition* network = builder->createNetwork();

  // Create the UFF parser
  IUffParser* parser = createUffParser();
  assert(parser);

  // Declare the network inputs and outputs of the model to the parser
  parser->registerInput(INPUT_TENSOR_NAME, DimsCHW(20, 16, 16), UffInputOrder::kNCHW);
  parser->registerOutput(OUTPUT_TENSOR_NAME);

  // Parse the imported model to populate the network
  parser->parse(fileName.c_str(), *network, nvinfer1::DataType::kFLOAT); // use kHALF for FP16

  std::cout << "*** IMPORTING DONE ***" << std::endl; 
    
  // *** BUILDING THE ENGINE ***
  std::cout << "*** BUILDING THE ENGINE ***" << std::endl;
    
  //Build the engine using the builder object
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(MAX_WORKSPACE);
  //builder->setFp16Mode(true); //16-bit kernels are permitted --this is useful on GPUs supporting full FP16 operations
  ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);
  std::cout << "*** BUILDING DONE ***" << std::endl; 

  // Destroy network, builder and parser
  network->destroy();
  builder->destroy();
  parser->destroy();

    
  // *** SERIALIZE THE ENGINE HERE IF NEEDED FOR LATER USE ***

    
  // *** PERFORMING INFERENCE ***
  std::cout << "*** PERFORMING INFERENCE ***" << std::endl;

  // Create the input and the output buffers on Host
  float *output = new float[batchSize * OUTPUT_SIZE];
  
  //Open the file and the dataset
  H5File file( FILE_NAME, H5F_ACC_RDONLY );
  DataSet dataset = file.openDataSet( DATASET_NAME_DATA );

  //Get dataspace of the dataset
  DataSpace dataspace = dataset.getSpace();

  //Get the number of dimensions in the dataspace
  int rank = dataspace.getSimpleExtentNdims();
  hsize_t dims[2];
  int status_n = dataspace.getSimpleExtentDims(dims, NULL);
  std::cout << "Rank: " << rank << "\n";
  std::cout << "Dimensions: " << dims[0] << " " << dims[1] << std::endl;

  //Define the memory space to read dataset
  DataSpace memspace(rank,dims);
  std::cout << "MEMSPACE CREATED" << std::endl;

  //Read dataset back and display
  float *input_data = new float[dims[0] * dims[1]];
  dataset.read(input_data, PredType::NATIVE_FLOAT, memspace, dataspace);
  std::cout << "DATASET READ" << std::endl;

  int i(0);
  //srand(time(NULL));
  //i = rand() % dims[0];
  std::cout << "Starting from Image number: " << i << std::endl;
    
  // Engine requires exactly IEngine::getNbBindings() number of buffers  
  int nbBindings = engine->getNbBindings();
  assert(nbBindings == 2); // 1 input and 1 output

  // Create streams
  int numstreams(5);
  cudaStream_t stream[numstreams];
  for (i = 0; i < numstreams; i++)
      CHECK(cudaStreamCreate(&stream[i]));

  // Create a context to store intermediate activation values for each stream
  IExecutionContext *context = engine->createExecutionContext();
  assert(context);
  
  // Initialize and compute variables needed for the streams
  int batchSize_perStream(0);
  batchSize_perStream = batchSize/numstreams;
  int sizeInputStream(0);
  sizeInputStream = batchSize_perStream * INPUT_CH * INPUT_H * INPUT_W;
  int sizeOutputStream(0);
  sizeOutputStream = batchSize_perStream * OUTPUT_SIZE;
    
  // number of pointers equal to the number of cudaStream (5)
  void *buffers[numstreams][nbBindings];

  const int inputIndex = engine->getBindingIndex(INPUT_TENSOR_NAME);
  const int outputIndex = engine->getBindingIndex(OUTPUT_TENSOR_NAME);


  // Create GPU buffers on device
  for(i = 0; i < numstreams; i++){
      CHECK(cudaMalloc(&buffers[i][inputIndex], sizeInputStream * sizeof(float)));
      CHECK(cudaMalloc(&buffers[i][outputIndex], sizeOutputStream * sizeof(float)));
  }
    
  // Copy the data from host to device
  auto t_start = std::chrono::high_resolution_clock::now();
  for(i = 0; i < numstreams; i++)
      CHECK(cudaMemcpyAsync(buffers[i][inputIndex], input_data + i * sizeInputStream, sizeInputStream * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
   
  // Enqueue the kernels on CUDA streams for the asynchronous execution
  for(i = 0; i < numstreams; i++)
      context->enqueue(batchSize_perStream, buffers[i], stream[i], nullptr);
 
  // Copy the data from device to host
  for (i = 0; i < numstreams; i++)
      CHECK(cudaMemcpyAsync(output + i * sizeOutputStream, buffers[i][outputIndex], sizeOutputStream * sizeof(float), cudaMemcpyDeviceToHost, stream[i]));
    
  // Synchronize
  cudaDeviceSynchronize();
  auto t_end = std::chrono::high_resolution_clock::now();
  ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
  
  //cudaStreamSynchronize(stream[i]);

  // Release buffers
  for (i = 0; i < numstreams; i++){
      CHECK(cudaStreamDestroy(stream[i]));
      CHECK(cudaFree(buffers[i][inputIndex]));
      CHECK(cudaFree(buffers[i][outputIndex]));
  }
    
  // Destroy the context and the engine
  engine->destroy();
  context->destroy();

  // Print the time of execution and histogram of the output distribution     
  std::cout << "\n*** OUTPUT ***\n\n" << std::endl;

  // Create a dataset for labels
  DataSet dataset_lb = file.openDataSet( DATASET_NAME_LABELS );
    
  //Get dataspace of the dataset
  DataSpace dataspace_lb = dataset_lb.getSpace();
    
  //Get the number of dimensions in the dataspace
  int rank_lb = dataspace_lb.getSimpleExtentNdims();
  hsize_t dims_lb[2];
  int status_n_lb = dataspace_lb.getSimpleExtentDims(dims_lb, NULL);
  std::cout << "Rank: " << rank_lb << "\n";
  std::cout << "Dimensions: " << dims_lb[0] << " " << dims_lb[1] << std::endl;
    
  //Define the memory space to read dataset
  DataSpace memspace_lb(rank_lb,dims_lb);
  std::cout << "MEMSPACE CREATED" << std::endl;
    
  //Read dataset back and display
  float *label_output = new float[dims_lb[0] * dims_lb[1]];
  dataset_lb.read(label_output, PredType::NATIVE_FLOAT, memspace_lb, dataspace_lb);
  std::cout << "LABEL DATASET READ" << std::endl;
    
  for(i = 0; i <  batchSize; i++){
    std::cout << "Image n: " << i << "\n";
    std::cout << "y_pred = (" << output[2*i] << "," << output[2*i+1] << ") | y_true = (" << label_output[2*i] << "," << label_output[2*i+1] << ")\n";
  }

  std::cout << "\nTime to perform inference: " << ms << "ms\n" << std::endl;
  
  delete[] input_data;
  delete[] output;
  delete[] label_output;

  shutdownProtobufLibrary();
  return 0;

}






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


#define MAX_WORKSPACE (1 << 30)
const int maxBatchSize = 1;

//define the data info
const H5std_string FILE_NAME("pixel_only_data_test.h5");
const H5std_string DATASET_NAME("data/block0_values");

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
  std::cout << "*** DATASET FILE: " << FILE_NAME << std::endl;

  int batchSize = 1;
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
  parser->parse(fileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

  std::cout << "*** IMPORTING DONE ***" << std::endl; 


  // *** BUILDING THE ENGINE ***
  std::cout << "*** BUILDING THE ENGINE ***" << std::endl;
  
  //Build the engine using the builder object
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(MAX_WORKSPACE);
  //builder->setFp16Mode(true); //16-bit kernels are permitted
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

    // Create a context to store intermediate activation values
  IExecutionContext *context = engine->createExecutionContext();
  assert(context);

  // Create the input and the output buffers on Host
  float input[INPUT_CH * INPUT_H * INPUT_W];
  float *input_data;
  float label_output[2];
  float output[OUTPUT_SIZE];
 
  //Create an input buffer to test a single row
  float *input_onerow;
 
  //Open the file and the dataset
  H5File file( FILE_NAME, H5F_ACC_RDONLY );
  DataSet dataset = file.openDataSet( DATASET_NAME );

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
  input_data = (float *) malloc(dims[0] * dims[1] * sizeof(float));
  dataset.read(input_data, PredType::NATIVE_FLOAT, memspace, dataspace);
  std::cout << "DATASET READ" << std::endl;

  int i, j;
  srand(time(NULL));
  i = rand() % dims[0];
  std::cout << "Image number: " << i << std::endl; 
  input_onerow = (float *) malloc((dims[1]-2) * sizeof(float));
  for(j = 0; j < dims[1] - 2; j++){
    input_onerow[j]=input_data[i * dims[1] + j];
    //std::cout << input_onerow[j] << " ";
  }
  //std::cout << "\n" << std::endl;
  label_output[0] = input_data[i*dims[1] + dims[1] - 2];
  label_output[1] = input_data[i*dims[1] + dims[1] - 1];
  std::cout << "Label output: y0 = " << label_output[0] << " y1 = " << label_output[1] << std::endl;


  // Engine requires exactly IEngine::getNbBindings() number of buffers  
  int nbBindings = engine->getNbBindings();
  assert(nbBindings == 2); // 1 input and 1 output
  
  void* buffers[nbBindings];

  const int inputIndex = engine->getBindingIndex(INPUT_TENSOR_NAME);
  const int outputIndex = engine->getBindingIndex(OUTPUT_TENSOR_NAME);

  
  // Create GPU buffers on device                                            
  CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_CH * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

  // Create stream                                                           
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Copy the data from host to device
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input_onerow, batchSize * INPUT_CH * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
  
  // Enqueue the kernels on a CUDA stream (TensorRT execution is typically asynchronous)
  auto t_start = std::chrono::high_resolution_clock::now();
  context->enqueue(batchSize, buffers, stream, nullptr);
  //context->execute(batchSize, buffers);
  cudaStreamSynchronize(stream); 
  auto t_end = std::chrono::high_resolution_clock::now();
  ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

  // Copy the data from device to host
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

  // Synchronize
  cudaStreamSynchronize(stream);

  // Release buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));

  // Destroy the context and the engine
  context->destroy();
  engine->destroy();

  // Print the time of execution and histogram of the output distribution     
  std::cout << "\n*** OUTPUT ***\n\n";
  std::cout << "Time to perform inference: " << ms << "ms\n" << std::endl;
  
  for (int i = 0; i < OUTPUT_SIZE; i++)
    std::cout << "y" << i << " = " << output[i] << "\n";

  std::cout << std::endl;

  shutdownProtobufLibrary();
  return 0;

}






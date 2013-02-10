#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "timer.h"
#include "utils.h"

#include "PyramidMaker.h"

int main(int argc, char** argv) {
  std::string input_file;
  std::string output_file;
  if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
  } else {
    std::cerr << "Usage: ./hw input_file output_file" << std::endl;
    return EXIT_FAILURE;
  }

  GpuTimer timer;
  timer.Start();
  MakePyramids(input_file, output_file);
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  printf("Time: %f ms\n", timer.Elapsed());

  return EXIT_SUCCESS;
}


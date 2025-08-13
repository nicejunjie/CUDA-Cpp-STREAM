/*
   STREAM benchmark implementation in CUDA.

COPY:       a(i) = b(i)
SCALE:      a(i) = q*b(i)
SUM:        a(i) = b(i) + c(i)
TRIAD:      a(i) = b(i) + q*c(i)

It measures the memory system on the device.
The implementation is in double precision.

Code based on the code developed by John D. McCalpin
http://www.cs.virginia.edu/stream/FTP/Code/stream.c

Written by: Massimiliano Fatica, NVIDIA Corporation

Further modifications by: Ben Cumming, CSCS; Andreas Herten (JSC/FZJ); Sebastian Achilles (JSC/FZJ)
 */

#ifdef NTIMES
#if NTIMES <= 1
#   define NTIMES  20
#endif
#endif
#ifndef NTIMES
#   define NTIMES  20
#endif

#include <string>
#include <vector>

#include <stdio.h>
#include <float.h>
#include <limits.h>
// #include <unistd.h>
#include <getopt.h>

#include <chrono>

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

typedef double real;

static double   avgtime[4] = {0}, maxtime[4] = {0},
                mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};


void print_help()
{
  printf(
      "Usage: stream [-s] [-c [-f]] [-n <elements>] [-b <blocksize>]\n\n"
      "  -s, --si\n"
      "        Print results in SI units (by default IEC units are used)\n\n"
      "  -c, --csv\n"
      "        Print results CSV formatted\n\n"
      "  -f, --full\n"
      "        Print all results in CSV\n\n"
      "  -t, --title\n"
      "        Print CSV header\n\n"
      "  -n <elements>, --nelements <element>\n"
      "        Put <elements> values in the arrays\n"
      "        (default: 1<<26)\n\n"
      "  -b <blocksize>, --blocksize <blocksize>\n"
      "        Use <blocksize> as the number of threads in each block\n"
      "        (default: 192)\n"
      );
}

void parse_options(int argc, char** argv, bool& SI, bool& CSV, bool& CSV_full, bool& CSV_header, int& N, int& blockSize)
{
  // Default values
  SI = false;
  CSV = false;
  CSV_full = false;
  CSV_header = false;
  N = 1<<26;
  blockSize = 192;

  static struct option long_options[] = {
    {"si",        no_argument,       0,  's' },
    {"csv",       no_argument,       0,  'c' },
    {"full",      no_argument,       0,  'f' },
    {"title",     no_argument,       0,  't' },
    {"nelements", required_argument, 0,  'n' },
    {"blocksize", required_argument, 0,  'b' },
    {"help",      no_argument,       0,  'h' },
    {0,           0,                 0,  0   }
  };
  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "scftn:b:h", long_options, &option_index)) != -1)
    switch (c)
    {
      case 's':
        SI = true;
        break;
      case 'c':
        CSV = true;
        break;
      case 'f':
        CSV_full = true;
        break;
      case 't':
        CSV_header = true;
        break;
      case 'n':
        N = std::atoi(optarg);
        break;
      case 'b':
        blockSize = std::atoi(optarg);
        break;
      case 'h':
        print_help();
        std::exit(0);
        break;
      default:
        print_help();
        std::exit(1);
    }
}

  template <typename T>
__global__ void set_array(T * __restrict__ const a, T value, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    a[idx] = value;
}

  template <typename T>
__global__ void STREAM_Copy(T const * __restrict__ const a, T * __restrict__ const b, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    b[idx] = a[idx];
}

  template <typename T>
__global__ void STREAM_Scale(T const * __restrict__ const a, T * __restrict__ const b, T scale,  int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    b[idx] = scale * a[idx];
}

  template <typename T>
__global__ void STREAM_Add(T const * __restrict__ const a, T const * __restrict__ const b, T * __restrict__ const c, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    c[idx] = a[idx] + b[idx];
}

  template <typename T>
__global__ void STREAM_Triad(T const * __restrict__ a, T const * __restrict__ b, T * __restrict__ const c, T scalar, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    c[idx] = a[idx] + scalar * b[idx];
}

int main(int argc, char** argv)
{
  real *d_a, *d_b, *d_c;
  int j,k;
  double times[4][NTIMES];
  real scalar;
  std::chrono::steady_clock::time_point start_time, end_time;
  std::vector<std::string> label{"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

  // Parse arguments
  bool SI, CSV, CSV_full, CSV_header;
  int N, blockSize;
  parse_options(argc, argv, SI, CSV, CSV_full, CSV_header, N, blockSize);

  if (!CSV) {
    printf("STREAM Benchmark implementation in CUDA\n");
    printf("Array size (%s precision) = %7.2f MB\n", sizeof(double)==sizeof(real)?"double":"single", double(N)*double(sizeof(real))/1.e6);
  }

  /* Allocate memory on device */
#ifdef USE_HOST
  d_a=(real*)malloc(sizeof(real)*N);
  d_b=(real*)malloc(sizeof(real)*N);
  d_c=(real*)malloc(sizeof(real)*N);
#elif defined(ZERO_COPY)
  real *h_a, *h_b, *h_c;
  cudaHostAlloc((void **) &h_a, sizeof(real)*N, cudaHostAllocMapped);
  cudaHostAlloc((void **) &h_b, sizeof(real)*N, cudaHostAllocMapped);
  cudaHostAlloc((void **) &h_c, sizeof(real)*N, cudaHostAllocMapped);

  // these compiles fine but don't run correctly.
  //h_a=(real*)malloc(sizeof(real)*N);
  //h_b=(real*)malloc(sizeof(real)*N);
  //h_c=(real*)malloc(sizeof(real)*N);

  cudaHostGetDevicePointer((void **) &d_a, (void *) h_a, 0);
  cudaHostGetDevicePointer((void **) &d_b, (void *) h_b, 0);
  cudaHostGetDevicePointer((void **) &d_c, (void *) h_c, 0);
#else 
  cudaMalloc((void**)&d_a, sizeof(real)*N);
  cudaMalloc((void**)&d_b, sizeof(real)*N);
  cudaMalloc((void**)&d_c, sizeof(real)*N);
#endif

  /* Compute execution configuration */
  dim3 dimBlock(blockSize);
  dim3 dimGrid(N/dimBlock.x );
  if( N % dimBlock.x != 0 ) dimGrid.x+=1;

  if (!CSV) {
    printf("Using %d threads per block, %d blocks\n",dimBlock.x,dimGrid.x);

    if (SI)
      printf("Output in SI units (KB = 1000 B)\n");
    else
      printf("Output in IEC units (KiB = 1024 B)\n");
  }

  /* Initialize memory on the device */
  set_array<real><<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<real><<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<real><<<dimGrid,dimBlock>>>(d_c, .5f, N);

  /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */

  scalar=3.0f;
  for (k=0; k<NTIMES; k++)
  {
    start_time = std::chrono::steady_clock::now();
    STREAM_Copy<real><<<dimGrid,dimBlock>>>(d_a, d_c, N);
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    times[0][k] = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    start_time = std::chrono::steady_clock::now();
    STREAM_Scale<real><<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    times[1][k] = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    start_time = std::chrono::steady_clock::now();
    STREAM_Add<real><<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    times[2][k] = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    start_time = std::chrono::steady_clock::now();
    STREAM_Triad<real><<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    times[3][k] = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
  }

  /*  --- SUMMARY --- */

  for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
  {
    for (j=0; j<4; j++)
    {
      avgtime[j] = avgtime[j] + times[j][k];
      mintime[j] = MIN(mintime[j], times[j][k]);
      maxtime[j] = MAX(maxtime[j], times[j][k]);
    }
  }
  for (j=0; j<4; j++)
    avgtime[j] = avgtime[j]/(double)(NTIMES-1);

  double bytes[4] = {
    2 * sizeof(real) * (double)N,
    2 * sizeof(real) * (double)N,
    3 * sizeof(real) * (double)N,
    3 * sizeof(real) * (double)N
  };

  // Use right units
  const double G = SI ? 1.e9 : static_cast<double>(1<<30);
  std::string gbpersec = SI ? "GB/s" : "GiB/s";

  if (!CSV) {
    printf("\nFunction      Rate %s  Avg time(s)  Min time(s)  Max time(s)\n", gbpersec.c_str() );
    printf("-----------------------------------------------------------------\n");
    for (j=0; j<4; j++) {
      printf("%s%11.2f     %11.8f  %11.8f  %11.8f\n", label[j].c_str(),
          bytes[j]/mintime[j] / G,
          avgtime[j],
          mintime[j],
          maxtime[j]);
    }
  } else {
    if (CSV_full) {
      if (CSV_header)
        printf("Copy (Max) / %s, Copy (Min) / %s, Copy (Avg) / %s, Scale (Max) / %s, Scale (Min) / %s, Scale (Avg) / %s, Add (Max) / %s, Add (Min) / %s, Add (Avg) / %s, Triad (Max) / %s, Triad (Min) / %s, Triad (Avg) / %s\n",
            gbpersec.c_str(), gbpersec.c_str(), gbpersec.c_str(),
            gbpersec.c_str(), gbpersec.c_str(), gbpersec.c_str(),
            gbpersec.c_str(), gbpersec.c_str(), gbpersec.c_str(),
            gbpersec.c_str(), gbpersec.c_str(), gbpersec.c_str()
            );
      printf(
          "%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f\n",
          bytes[0]/mintime[0] / G, bytes[0]/maxtime[0] / G, bytes[0]/(avgtime[0])/ G,
          bytes[1]/mintime[1] / G, bytes[1]/maxtime[1] / G, bytes[1]/(avgtime[1]) / G,
          bytes[2]/mintime[2] / G, bytes[2]/maxtime[2] / G, bytes[2]/(avgtime[2]) / G,
          bytes[3]/mintime[3] / G, bytes[3]/maxtime[3] / G, bytes[3]/(avgtime[3]) / G
          );
    }
    else {
      if (CSV_header)
        printf("Copy (Max) / %s, Scale (Max) / %s, Add (Max) / %s, Triad (Max) / %s\n", gbpersec.c_str(), gbpersec.c_str(), gbpersec.c_str(), gbpersec.c_str());
      printf(
          "%0.4f,%0.4f,%0.4f,%0.4f\n",
          bytes[0]/mintime[0] / G,
          bytes[1]/mintime[1] / G,
          bytes[2]/mintime[2] / G,
          bytes[3]/mintime[3] / G
          );
    }
  }


  /* Free memory on device */
#ifdef USE_HOST
  free(d_a);
  free(d_b);
  free(d_c);
#elif defined(ZERO_COPY)
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
//  free(h_a);
//  free(h_b);
//  free(h_c);
#else
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
#endif
}


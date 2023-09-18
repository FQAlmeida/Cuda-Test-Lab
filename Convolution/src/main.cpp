#include <benchmark/benchmark.h>

#include "convolution/convolution.cuh"
#include "convolution/convolution_serial.hpp"
#include "convolution/convolution_cpu_par.hpp"

static void BM_convolution_par(benchmark::State& state) {
    // Perform setup here
    for (auto _ : state) {
        // This code gets timed
        run_convolution(40);
    }
}
// Register the function as a benchmark
BENCHMARK(BM_convolution_par);
static void BM_convolution_searial(benchmark::State& state) {
    // Perform setup here
    for (auto _ : state) {
        // This code gets timed
        run_convolution_serial(40);
    }
}
// Register the function as a benchmark
BENCHMARK(BM_convolution_searial);
static void BM_convolution_cpu_par(benchmark::State& state) {
    // Perform setup here
    for (auto _ : state) {
        // This code gets timed
        run_convolution_par(40);
    }
}
// Register the function as a benchmark
BENCHMARK(BM_convolution_cpu_par);
// Run the benchmark
// BENCHMARK_MAIN();

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}

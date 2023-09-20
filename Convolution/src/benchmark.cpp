#include <benchmark/benchmark.h>

#include "convolution/convolution.cuh"
#include "convolution/convolution_serial.hpp"
#include "convolution/convolution_cpu_par.hpp"

static void BM_convolution_par(benchmark::State& state) {
    for (auto _ : state) {
        run_convolution(20, 1000);
    }
}
BENCHMARK(BM_convolution_par);

static void BM_convolution_cpu_par(benchmark::State& state) {
    for (auto _ : state) {
        run_convolution_par(20, 1000);
    }
}
BENCHMARK(BM_convolution_cpu_par);

static void BM_convolution_serial(benchmark::State& state) {
    for (auto _ : state) {
        run_convolution_serial(20, 1000);
    }
}
BENCHMARK(BM_convolution_serial);

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}

#include <benchmark/benchmark.h>

#include "convolution/convolution.cuh"
#include "convolution/convolution_cpu_par.hpp"
#include "convolution/convolution_serial.hpp"

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int k = 1; k <= 11; k++) {
        b->Args({10, 1, k});
    }
    for (int k = 1; k <= 11; k++) {
        b->Args({10, 100, k});
    }
    for (int k = 1; k <= 11; k++) {
        b->Args({50, 1, k});
    }
    for (int k = 1; k <= 11; k++) {
        b->Args({50, 100, k});
    }
}

static void BM_convolution_par(benchmark::State& state) {
    for (auto _ : state) {
        run_convolution(state.range(0), state.range(1), state.range(2));
    }
}
BENCHMARK(BM_convolution_par)->Iterations(5)->Apply(CustomArguments);

static void BM_convolution_cpu_par(benchmark::State& state) {
    for (auto _ : state) {
        run_convolution_par(state.range(0), state.range(1), state.range(2));
    }
}
BENCHMARK(BM_convolution_cpu_par)->Iterations(5)->Apply(CustomArguments);

static void BM_convolution_serial(benchmark::State& state) {
    for (auto _ : state) {
        run_convolution_serial(state.range(0), state.range(1), state.range(2));
    }
}
// BENCHMARK(BM_convolution_serial)->Iterations(3)->Apply(CustomArguments);

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}

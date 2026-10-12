<p align="center">
  <img src="assets/parallel-canny-banner.png"
       alt="Parallel Canny — C++ POSIX threads image processing"
       width="100%">
</p>

### Seeing edges differently — through C++, pthreads, and systems thinking.

<p align="center">
  <a href="https://github.com/solar224/canny-edge-detector-parallel"><img src="https://img.shields.io/badge/study%20of-solar224%2Fcanny--edge--detector--parallel-1c2128?style=flat-square" alt="Study of solar224/canny-edge-detector-parallel"></a>
  <img src="https://img.shields.io/badge/C%2B%2B-17-1c2128?style=flat-square" alt="C++17">
  <img src="https://img.shields.io/badge/POSIX-threads-1c2128?style=flat-square" alt="POSIX threads">
  <img src="https://img.shields.io/badge/license-MIT-1c2128?style=flat-square" alt="MIT License">
</p>

<p align="center">
  <sub>C++ &nbsp;•&nbsp; Linux &nbsp;•&nbsp; POSIX Threads &nbsp;•&nbsp; OpenCV &nbsp;•&nbsp; Image Processing &nbsp;•&nbsp; Concurrency</sub>
</p>

I wanted to understand what happens when an image-processing algorithm stops being just an algorithm and becomes a systems problem. Canny edge detection was a concrete place to look: the stages are familiar, but a pthread version has to decide who owns each row, which writes other threads may see, and which stage cannot be split. This repository is my study of that pipeline — memory, threads, synchronization, and the dependencies the algorithm does not give up.

## Project overview

This is a study of a C++ Canny detector parallelized with POSIX threads. I did not write it. The serial path is [`canny.cpp`](canny.cpp); [`main.cpp`](main.cpp) calls the pthread path in [`canny_parallel.cpp`](canny_parallel.cpp). OpenCV reads and writes the file. The filters are in this tree.

I used the project to study multithreading, synchronization, shared memory, image processing, performance tradeoffs, and concurrency correctness: which stages are safe because of the split, and which only look safe.

## From pixels to edges

```
Input → Gaussian Blur → Grayscale → Sobel Gradients → Non-Maximum Suppression → Hysteresis → Edge Map
```

`cannyEdgeDetection_parallel` runs those stages in order. Each one finishes before the next starts.

- **Gaussian blur.** A fixed 5×5 kernel, scaled by `1/159`, runs per channel. Off-image samples are dropped, and the pixel is divided by the weights that were used.
- **Grayscale.** Channels are averaged into one integer per pixel.
- **Sobel gradients.** A 3×3 `gx` / `gy` pair produces a magnitude and an angle snapped to a multiple of 45°, and tracks the largest magnitude.
- **Non-maximum suppression.** Along that direction, a weaker pixel is cleared. Survivors are scaled by `255 / largestG`.
- **Hysteresis.** Values below `lowerThreshold * largestG` are cleared, values at or above `higherThreshold * largestG` stay, and the rest are promoted if an 8-neighbor is already strong. The scan repeats until a pass changes nothing. `main` uses `0.03` and `0.1`.

| Original | Edge Detection |
|---|---|
| ![](assets/original.jpg) | ![](assets/edges.jpg) |

## Where the parallelism happens

The split is static row partitioning. Each parallel function sets `rowsPerThread = sizeRows / numThreads` and hands every worker a `startRow` and `endRow`. The last thread’s `endRow` is `sizeRows`, so it takes the leftover rows. `setNumThreads` clamps the count to 1–16.

```
Image
├── Thread 0 → Rows 0–N
├── Thread 1 → Rows N–2N
├── Thread 2 → Rows 2N–3N
└── Thread 3 → Remaining rows
```

`gaussianBlurWorker`, `rgbToGrayscaleWorker`, `cannyPhase1Worker` (Sobel), and `cannyPhase2Worker` (suppression) share that layout. Blur, grayscale, and Sobel do not lock pixel stores. Each worker writes only its own rows, and a read that crosses the boundary comes from the previous buffer, which that stage does not modify.

Suppression is started the same way, then writes the magnitude array it is also reading. That is the break in the pattern.

## Synchronization

There is no thread pool. `gaussianBlur_parallel`, `rgbToGrayscale_parallel`, and `cannyFilter_parallel` each call `pthread_create`, then `pthread_join` on every worker. The Canny filter does this twice: phase 1 for gradients, phase 2 for suppression.

`pthread_join` is the synchronization point between stages. Worker stores happen-before the join returns, so the next stage can read those pixels without a lock. Border fill runs on the caller after phase 1. Hysteresis runs there after phase 2.

The only mutex protects a scalar. Each Sobel worker keeps a private `localLargestG`, then takes `pthread_mutex_lock` once to merge it into the shared `largestG`. Phase 2 reads that value after the joins, which have already published it.

`ThreadData` includes a `pthread_barrier_t* barrier`. Nothing initializes it, assigns it, or calls `pthread_barrier_wait`.

## The interesting part: finding a concurrency problem

> ### ⚠️ A subtle race inside non-maximum suppression
>
> Suppression reads a neighbor’s magnitude from `G` and, in the same pass, writes `0` back into `G`.

Horizontal neighbors sit on this thread’s row. Vertical and diagonal neighbors do not:

```cpp
if (currentG < data->G[(i + 1) * data->sizeCols + j + 1] ||
    currentG < data->G[(i - 1) * data->sizeCols + j - 1]) {
    data->G[i * data->sizeCols + j] = 0;
}
```

Row `i + 1` can belong to another worker that is storing `0` into the cell being loaded. Nothing synchronizes those accesses, so two runs of the same image can disagree.

Blur and Sobel are safe because their input is frozen. Suppression uses `G` as both stencil and destination.

The design change is to read the magnitudes Sobel wrote and store suppressed values in a separate buffer. Neighbor loads then cannot race, and the edges no longer depend on scan order. That fix is not in this tree. The code still writes the zeros back into `G`.

## Performance: where the time really goes

This is the shape of the code, not a timing I measured.

`imgToArray` widens `uint8` pixels into a `std::vector<int>`. Blur allocates another full RGB integer frame. Sobel adds a `double` array for `G` and an `int` array for `theta`. Every stage reads one whole frame and writes another before the next stage may start, so much of the work is moving wide buffers. Grayscale does very little arithmetic per byte it touches.

On top of that traffic, the parallel path calls `pthread_create` and `pthread_join` four times per image, and `atan2` runs on every interior pixel. Hysteresis stays outside the split. One thread scans the frame, and a weak chain against row-major order grows by about a pixel each pass.

More threads can shorten the row stencils. Under Amdahl’s Law they do not shorten conversion, border fill, thread spawn, or hysteresis. A persistent pool, narrower samples, cache-sized tiles with a halo, direction tests on `|gx|` and `|gy|` instead of `atan2`, and double-buffered suppression would attack those costs. None of that is implemented. [`benchmark.cpp`](benchmark.cpp) only times the stages that exist, over 10 runs for 1–6 threads.

## Why hysteresis stays serial

Earlier stages are stencils: the write does not change what the next pixel must read from that buffer. Hysteresis propagates. A weak pixel is cleared, then set to `higherThreshold * largestG` if any 8-neighbor is already strong, and that store can qualify a pixel on another row. The loop repeats while `changes` is set because one row-major pass cannot finish every chain.

A parallel version has to change the algorithm — double-buffered generations, or connectivity from strong seeds. After phase 2 joins, `cannyFilter_parallel` does neither. The loop stays on the calling thread.

## Build and run

CMake 3.18+, C++17, OpenCV, and pthreads. The Ubuntu workflow in this repo builds with:

```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev build-essential

mkdir build
cd build
cmake ..
make -j$(nproc)
```

That links `canny` and `benchmark` against `canny.cpp` and `canny_parallel.cpp`. With no arguments, `canny` uses one thread, reads `../images/Sukuna.jpg`, and writes `../images/SukunaCanny.jpg`. The count is clamped to 1–16. Input and output paths must differ.

```bash
./canny
./canny 4
./canny 4 ../images/Sukuna.jpg ../images/SukunaCanny.jpg

./benchmark
./benchmark ../images/Sukuna.jpg
```

## What I took away

The threads are the obvious part. What changed my reading was asking which buffer is frozen, which address another core might be storing, and which loop stayed serial because the dependence does not split.

- Parallelism is not thread safety. Disjoint rows make blur and Sobel safe. In-place suppression does not inherit that safety.
- Synchronization should be deliberate. One mutex covers a max. Join orders the stages.
- Memory access can matter as much as computation. Wide full-frame integers are part of the cost.
- Some algorithms need a redesign before they parallelize. Hysteresis is one of them.
- Understanding the code means questioning its assumptions: the unused barrier, the shared `G`, the scan left after the workers join.

## Original source and attribution

This repository studies an existing implementation. It is not a claim that I authored the detector.

- **Original repository:** [solar224/canny-edge-detector-parallel](https://github.com/solar224/canny-edge-detector-parallel)
- **Copyright:** Copyright (c) 2026 solar224
- **License:** [MIT](LICENSE)

The MIT terms in [`LICENSE`](LICENSE) apply. The copyright notice and the permission notice must be included in any copy or substantial portion of the software.

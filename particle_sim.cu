#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WIDTH 100.0f
#define HEIGHT 100.0f

// Error checking macro
#define CHECK(stmt)                                      \
  do {                                                   \
    cudaError_t err = stmt;                              \
    if (err != cudaSuccess) {                            \
      std::cerr << "CUDA Error: "                        \
                << cudaGetErrorString(err) << std::endl; \
      exit(1);                                           \
    }                                                    \
  } while (0)

// ---------------- CPU VERSION ----------------
void update_particles_cpu(float *x, float *y,
                          float *vx, float *vy,
                          int numElements, float dt) {

    for (int i = 0; i < numElements; i++) {
        // gravity
        vy[i] += -9.8f * dt;

        // update position
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;

        // boundary collisions
        if (x[i] < 0 || x[i] > WIDTH)
            vx[i] *= -1;

        if (y[i] < 0 || y[i] > HEIGHT)
            vy[i] *= -1;
    }
}

// ---------------- GPU KERNEL ----------------
__global__ void update_particles_gpu(float *x, float *y,
                                     float *vx, float *vy,
                                     int numElements, float dt) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numElements) {
        // gravity
        vy[i] += -9.8f * dt;

        // update position
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;

        // boundary collisions
        if (x[i] < 0 || x[i] > WIDTH)
            vx[i] *= -1;

        if (y[i] < 0 || y[i] > HEIGHT)
            vy[i] *= -1;
    }
}

// ---------------- MAIN ----------------
int main() {

    int numElements = 1 << 16;  // 65536 particles
    float dt = 0.01f;

    // Host memory
    float *hostX = (float *)malloc(numElements * sizeof(float));
    float *hostY = (float *)malloc(numElements * sizeof(float));
    float *hostVX = (float *)malloc(numElements * sizeof(float));
    float *hostVY = (float *)malloc(numElements * sizeof(float));

    // Initialize particles
    for (int i = 0; i < numElements; i++) {
        hostX[i] = rand() % 100;
        hostY[i] = rand() % 100;
        hostVX[i] = rand() % 10 - 5;
        hostVY[i] = rand() % 10 - 5;
    }

    // ---------------- CPU RUN ----------------
    float *cpuX = (float *)malloc(numElements * sizeof(float));
    float *cpuY = (float *)malloc(numElements * sizeof(float));
    float *cpuVX = (float *)malloc(numElements * sizeof(float));
    float *cpuVY = (float *)malloc(numElements * sizeof(float));

    memcpy(cpuX, hostX, numElements * sizeof(float));
    memcpy(cpuY, hostY, numElements * sizeof(float));
    memcpy(cpuVX, hostVX, numElements * sizeof(float));
    memcpy(cpuVY, hostVY, numElements * sizeof(float));

    auto startCPU = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < 100; t++) {
        update_particles_cpu(cpuX, cpuY, cpuVX, cpuVY, numElements, dt);
    }

    auto endCPU = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

    std::cout << "CPU Time: " << cpuTime << " ms" << std::endl;

    // ---------------- GPU SETUP ----------------
    float *deviceX, *deviceY, *deviceVX, *deviceVY;

    CHECK(cudaMalloc((void **)&deviceX, numElements * sizeof(float)));
    CHECK(cudaMalloc((void **)&deviceY, numElements * sizeof(float)));
    CHECK(cudaMalloc((void **)&deviceVX, numElements * sizeof(float)));
    CHECK(cudaMalloc((void **)&deviceVY, numElements * sizeof(float)));

    CHECK(cudaMemcpy(deviceX, hostX, numElements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceY, hostY, numElements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceVX, hostVX, numElements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceVY, hostVY, numElements * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    // ---------------- GPU RUN ----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int t = 0; t < 100; t++) {
        update_particles_gpu<<<dimGrid, dimBlock>>>(
            deviceX, deviceY, deviceVX, deviceVY, numElements, dt);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;

    CHECK(cudaDeviceSynchronize());

    // Copy back results
    CHECK(cudaMemcpy(hostX, deviceX, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostY, deviceY, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "First particle position: "
              << hostX[0] << ", " << hostY[0] << std::endl;

    // Cleanup
    cudaFree(deviceX);
    cudaFree(deviceY);
    cudaFree(deviceVX);
    cudaFree(deviceVY);

    free(hostX);
    free(hostY);
    free(hostVX);
    free(hostVY);

    free(cpuX);
    free(cpuY);
    free(cpuVX);
    free(cpuVY);

    return 0;
}

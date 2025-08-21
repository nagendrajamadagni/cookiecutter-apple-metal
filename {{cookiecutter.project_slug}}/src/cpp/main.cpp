#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/AutoreleasePoolGuard.hpp>
#include <Metal/MetalBuffer.hpp>
#include <Metal/MetalContext.hpp>
#include <iostream>
#include <vector>

void printDeviceInfo(MTL::Device *device) {
    if (!device) {
        std::cout << "Device: NULL" << std::endl;
        return;
    }

    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Name: " << device->name()->utf8String() << std::endl;
    std::cout << "Registry ID: " << device->registryID() << std::endl;
    std::cout << "Headless: " << (device->isHeadless() ? "Yes" : "No")
              << std::endl;
    std::cout << "Unified Memory: "
              << (device->hasUnifiedMemory() ? "Yes" : "No") << std::endl;

    // Memory information
    std::cout << "Recommended Max Working Set Size: "
              << (device->recommendedMaxWorkingSetSize() / (1024 * 1024))
              << " MB" << std::endl;

    // Feature support - test common GPU families
    std::cout << "Supports Family Mac 1: "
              << (device->supportsFamily(MTL::GPUFamilyMac1) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Mac 2: "
              << (device->supportsFamily(MTL::GPUFamilyMac2) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 1: "
              << (device->supportsFamily(MTL::GPUFamilyApple1) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 2: "
              << (device->supportsFamily(MTL::GPUFamilyApple2) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 3: "
              << (device->supportsFamily(MTL::GPUFamilyApple3) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 4: "
              << (device->supportsFamily(MTL::GPUFamilyApple4) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 5: "
              << (device->supportsFamily(MTL::GPUFamilyApple5) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 6: "
              << (device->supportsFamily(MTL::GPUFamilyApple6) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 7: "
              << (device->supportsFamily(MTL::GPUFamilyApple7) ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Family Apple 8: "
              << (device->supportsFamily(MTL::GPUFamilyApple8) ? "Yes" : "No")
              << std::endl;

    // Thread execution limits
    std::cout << "Max Threads Per Threadgroup: "
              << device->maxThreadsPerThreadgroup().width << std::endl;
    std::cout << "Max Buffer Length: "
              << (device->maxBufferLength() / (1024 * 1024)) << " MB"
              << std::endl;

    // Feature set queries
    std::cout << "Supports 32-bit Float Filtering: "
              << (device->supports32BitFloatFiltering() ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Query Texture LOD: "
              << (device->supportsQueryTextureLOD() ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports BC Texture Compression: "
              << (device->supportsBCTextureCompression() ? "Yes" : "No")
              << std::endl;
    std::cout << "Supports Pull Model Interpolation: "
              << (device->supportsPullModelInterpolation() ? "Yes" : "No")
              << std::endl;

    std::cout << "=========================" << std::endl << std::endl;
}

void printAllDevices() {
    std::cout << "\n=== All Available Metal Devices ===" << std::endl;

    NS::Array *devices = MTL::CopyAllDevices();
    if (!devices || devices->count() == 0) {
        std::cout << "No Metal devices found!" << std::endl;
        return;
    }

    std::cout << "Found " << devices->count()
              << " Metal device(s):" << std::endl
              << std::endl;

    for (NS::UInteger i = 0; i < devices->count(); i++) {
        MTL::Device *device = static_cast<MTL::Device *>(devices->object(i));
        std::cout << "Device " << (i + 1) << ":" << std::endl;
        printDeviceInfo(device);
    }

    devices->release();
}

int main() {
    // Initialize an autorelease pool guard
    AutoreleasePoolGuard guard;

    MetalContext context("build/{{cookiecutter.project_slug}}_kernel.metallib",
                         "vector_add");

    // Print information about all available Metal devices
    printAllDevices();

    std::cout << "Using default device for computation:" << std::endl;
    printDeviceInfo(context.getDevice().get());

    std::cout << "Performing a simple parallelized vector addition to test "
                 "working of GPU."
              << std::endl;

    // Prepare data
    const size_t count = 8;
    std::vector<float> A(count, 1.0f), B(count, 1.0f), C(count);

    // Create buffers on GPU

    MetalBuffer bufA(context, sizeof(float) * count);
    MetalBuffer bufB(context, sizeof(float) * count);
    MetalBuffer bufC(context, sizeof(float) * count);

    // Copy data into GPU buffers

    bufA.fillBuffer(A.data(), sizeof(float) * count);
    bufB.fillBuffer(B.data(), sizeof(float) * count);

    // Set buffer offset and index for kernel function

    context.setBuffer(bufA, 0, 0);
    context.setBuffer(bufB, 0, 1);
    context.setBuffer(bufC, 0, 2);

    // Set the dimensions for the threads that must run

    MetalDim gridDim(count, 1, 1);
    MetalDim blockDim(std::min((size_t)32, count), 1, 1);

    // Run the kernel with those dimensions

    context.runKernel(gridDim, blockDim);

    // Copy data back from GPU buffer to host buffer

    memcpy(C.data(), bufC.contents(), sizeof(float) * count);

    std::cout << "A = ";
    for (auto v : A)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "B = ";
    for (auto v : B)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "C = A + B = ";
    for (auto v : C)
        std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}

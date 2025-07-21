#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>

void printDeviceInfo(MTL::Device* device) {
    if (!device) {
        std::cout << "Device: NULL" << std::endl;
        return;
    }
    
    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Name: " << device->name()->utf8String() << std::endl;
    std::cout << "Registry ID: " << device->registryID() << std::endl;
    std::cout << "Headless: " << (device->isHeadless() ? "Yes" : "No") << std::endl;
    std::cout << "Unified Memory: " << (device->hasUnifiedMemory() ? "Yes" : "No") << std::endl;
    
    // Memory information
    std::cout << "Recommended Max Working Set Size: " << (device->recommendedMaxWorkingSetSize() / (1024 * 1024)) << " MB" << std::endl;
    
    // Feature support - test common GPU families
    std::cout << "Supports Family Mac 1: " << (device->supportsFamily(MTL::GPUFamilyMac1) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Mac 2: " << (device->supportsFamily(MTL::GPUFamilyMac2) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 1: " << (device->supportsFamily(MTL::GPUFamilyApple1) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 2: " << (device->supportsFamily(MTL::GPUFamilyApple2) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 3: " << (device->supportsFamily(MTL::GPUFamilyApple3) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 4: " << (device->supportsFamily(MTL::GPUFamilyApple4) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 5: " << (device->supportsFamily(MTL::GPUFamilyApple5) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 6: " << (device->supportsFamily(MTL::GPUFamilyApple6) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 7: " << (device->supportsFamily(MTL::GPUFamilyApple7) ? "Yes" : "No") << std::endl;
    std::cout << "Supports Family Apple 8: " << (device->supportsFamily(MTL::GPUFamilyApple8) ? "Yes" : "No") << std::endl;
    
    // Thread execution limits
    std::cout << "Max Threads Per Threadgroup: " << device->maxThreadsPerThreadgroup().width << std::endl;
    std::cout << "Max Buffer Length: " << (device->maxBufferLength() / (1024 * 1024)) << " MB" << std::endl;
    
    // Feature set queries
    std::cout << "Supports 32-bit Float Filtering: " << (device->supports32BitFloatFiltering() ? "Yes" : "No") << std::endl;
    std::cout << "Supports Query Texture LOD: " << (device->supportsQueryTextureLOD() ? "Yes" : "No") << std::endl;
    std::cout << "Supports BC Texture Compression: " << (device->supportsBCTextureCompression() ? "Yes" : "No") << std::endl;
    std::cout << "Supports Pull Model Interpolation: " << (device->supportsPullModelInterpolation() ? "Yes" : "No") << std::endl;
    
    std::cout << "=========================" << std::endl << std::endl;
}

void printAllDevices() {
    std::cout << "\n=== All Available Metal Devices ===" << std::endl;
    
    NS::Array* devices = MTL::CopyAllDevices();
    if (!devices || devices->count() == 0) {
        std::cout << "No Metal devices found!" << std::endl;
        return;
    }
    
    std::cout << "Found " << devices->count() << " Metal device(s):" << std::endl << std::endl;
    
    for (NS::UInteger i = 0; i < devices->count(); i++) {
        MTL::Device* device = static_cast<MTL::Device*>(devices->object(i));
        std::cout << "Device " << (i + 1) << ":" << std::endl;
        printDeviceInfo(device);
    }
    
    devices->release();
}

int main() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    
    // Print information about all available Metal devices
    printAllDevices();
    
    // Get the default device for computation
    auto* device = MTL::CreateSystemDefaultDevice();
    
    if (!device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return -1;
    }
    
    std::cout << "Using default device for computation:" << std::endl;
    printDeviceInfo(device);

    std::cout << "Performing a simple parallelized vector addition to test working of GPU." << std::endl;

    // Prepare data
    const size_t count = 8;
    std::vector<float> A(count, 1.0f), B(count, 1.0f), C(count);

    // Create buffers on GPU
    auto* bufA = device->newBuffer(A.data(), sizeof(float) * count, MTL::ResourceStorageModeManaged);
    auto* bufB = device->newBuffer(B.data(), sizeof(float) * count, MTL::ResourceStorageModeManaged);
    auto* bufC = device->newBuffer(sizeof(float) * count, MTL::ResourceStorageModeManaged);

    // Load shader library - try default library first, then metallib file
    NS::Error* error = nullptr;
    MTL::Library* lib = nullptr;
    
    // First try to load from default library (if compiled into app)
    lib = device->newDefaultLibrary();
    
    // If that fails, try to load from metallib file
    if (!lib) {
        auto* path = NS::String::string("build/{{cookiecutter.project_name}}_kernel.metallib", NS::UTF8StringEncoding);
        lib = device->newLibrary(path, &error);
    }
    
    if (!lib) {
        std::cerr << "Failed to load Metal library";
        if (error) {
            std::cerr << ": " << error->localizedDescription()->utf8String();
        }
        std::cerr << std::endl;
        device->release();
        pool->release();
        return -1;
    }

    auto* functionName = NS::String::string("vector_add", NS::UTF8StringEncoding);
    auto* fn = lib->newFunction(functionName);
    
    if (!fn) {
        std::cerr << "Failed to find function 'vector_add' in library" << std::endl;
        lib->release();
        device->release();
        pool->release();
        return -1;
    }

    auto* pipeline = device->newComputePipelineState(fn, &error);
    
    if (!pipeline) {
        std::cerr << "Failed to create compute pipeline state";
        if (error) {
            std::cerr << ": " << error->localizedDescription()->utf8String();
        }
        std::cerr << std::endl;
        fn->release();
        lib->release();
        device->release();
        pool->release();
        return -1;
    }

    // Encode and dispatch
    auto* queue = device->newCommandQueue();
    auto* cmdBuf = queue->commandBuffer();
    auto* enc = cmdBuf->computeCommandEncoder();

    enc->setComputePipelineState(pipeline);
    enc->setBuffer(bufA, 0, 0);
    enc->setBuffer(bufB, 0, 1);
    enc->setBuffer(bufC, 0, 2);

    // Thread configuration - use a reasonable threadgroup size
    MTL::Size grid(count, 1, 1);
    MTL::Size threadsPerThreadgroup(std::min((size_t)32, count), 1, 1); // Better threadgroup size
    
    enc->dispatchThreads(grid, threadsPerThreadgroup);
    enc->endEncoding();

    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    // Check for command buffer errors
    if (cmdBuf->status() == MTL::CommandBufferStatusError) {
        std::cerr << "Command buffer execution failed" << std::endl;
        if (cmdBuf->error()) {
            std::cerr << "Error: " << cmdBuf->error()->localizedDescription()->utf8String() << std::endl;
        }
    }

    // Get results back
    memcpy(C.data(), bufC->contents(), sizeof(float) * count);

    std::cout << "A = ";
    for (auto v : A) std::cout << v << " ";
    std::cout << std::endl;
    
    std::cout << "B = ";
    for (auto v : B) std::cout << v << " ";
    std::cout << std::endl;
    
    std::cout << "C = A + B = ";
    for (auto v : C) std::cout << v << " ";
    std::cout << std::endl;

    // Cleanup
    bufA->release();
    bufB->release();
    bufC->release();
    pipeline->release();
    fn->release();
    lib->release();
    queue->release();
    device->release();
    pool->release();

    return 0;
}

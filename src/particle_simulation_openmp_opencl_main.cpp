#include <CL/cl.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Particle {
    float x, y;
    float vx, vy;
};

struct Params {
    int width = 1280;
    int height = 720;
    int numParticles = 200000;
    int steps = 300;
    float dt = 1.0f;
    float gravityCenterX = 0.5f;
    float gravityCenterY = 0.5f;
    float attraction = 0.003f;
    float damping = 0.9995f;
    float bounce = 0.98f;
    int outputEvery = 100;
};

static double nowSeconds() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static std::vector<Particle> makeParticles(int n, int width, int height) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> px(0.0f, static_cast<float>(width));
    std::uniform_real_distribution<float> py(0.0f, static_cast<float>(height));
    std::uniform_real_distribution<float> pv(-2.0f, 2.0f);

    std::vector<Particle> particles(n);
    for (auto &p : particles) {
        p.x = px(rng);
        p.y = py(rng);
        p.vx = pv(rng);
        p.vy = pv(rng);
    }
    return particles;
}

static void stepSequential(std::vector<Particle> &particles, const Params &par) {
    const float cx = par.width * par.gravityCenterX;
    const float cy = par.height * par.gravityCenterY;

    for (auto &p : particles) {
        float dx = cx - p.x;
        float dy = cy - p.y;
        float dist2 = dx * dx + dy * dy + 1e-3f;
        float invDist = 1.0f / std::sqrt(dist2);
        float ax = dx * invDist * par.attraction;
        float ay = dy * invDist * par.attraction;

        p.vx = (p.vx + ax) * par.damping;
        p.vy = (p.vy + ay) * par.damping;
        p.x += p.vx * par.dt;
        p.y += p.vy * par.dt;

        if (p.x < 0.0f) {
            p.x = 0.0f;
            p.vx = -p.vx * par.bounce;
        } else if (p.x > par.width) {
            p.x = static_cast<float>(par.width);
            p.vx = -p.vx * par.bounce;
        }
        if (p.y < 0.0f) {
            p.y = 0.0f;
            p.vy = -p.vy * par.bounce;
        } else if (p.y > par.height) {
            p.y = static_cast<float>(par.height);
            p.vy = -p.vy * par.bounce;
        }
    }
}

static void stepOpenMP(std::vector<Particle> &particles, const Params &par) {
    const float cx = par.width * par.gravityCenterX;
    const float cy = par.height * par.gravityCenterY;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        auto &p = particles[i];
        float dx = cx - p.x;
        float dy = cy - p.y;
        float dist2 = dx * dx + dy * dy + 1e-3f;
        float invDist = 1.0f / std::sqrt(dist2);
        float ax = dx * invDist * par.attraction;
        float ay = dy * invDist * par.attraction;

        p.vx = (p.vx + ax) * par.damping;
        p.vy = (p.vy + ay) * par.damping;
        p.x += p.vx * par.dt;
        p.y += p.vy * par.dt;

        if (p.x < 0.0f) {
            p.x = 0.0f;
            p.vx = -p.vx * par.bounce;
        } else if (p.x > par.width) {
            p.x = static_cast<float>(par.width);
            p.vx = -p.vx * par.bounce;
        }
        if (p.y < 0.0f) {
            p.y = 0.0f;
            p.vy = -p.vy * par.bounce;
        } else if (p.y > par.height) {
            p.y = static_cast<float>(par.height);
            p.vy = -p.vy * par.bounce;
        }
    }
}

static std::string clErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "Unknown OpenCL error";
    }
}

static bool stepOpenCL(std::vector<Particle> &particles, const Params &par) {
    const char *kernelSource = R"CLC(
        typedef struct {
            float x, y;
            float vx, vy;
        } Particle;

        __kernel void step_particles(__global Particle* particles,
                                     const int width,
                                     const int height,
                                     const float centerX,
                                     const float centerY,
                                     const float attraction,
                                     const float damping,
                                     const float bounce,
                                     const float dt) {
            int id = get_global_id(0);
            Particle p = particles[id];

            float cx = width * centerX;
            float cy = height * centerY;
            float dx = cx - p.x;
            float dy = cy - p.y;
            float dist2 = dx * dx + dy * dy + 1e-3f;
            float invDist = native_rsqrt(dist2);
            float ax = dx * invDist * attraction;
            float ay = dy * invDist * attraction;

            p.vx = (p.vx + ax) * damping;
            p.vy = (p.vy + ay) * damping;
            p.x += p.vx * dt;
            p.y += p.vy * dt;

            if (p.x < 0.0f) {
                p.x = 0.0f;
                p.vx = -p.vx * bounce;
            } else if (p.x > width) {
                p.x = width;
                p.vx = -p.vx * bounce;
            }
            if (p.y < 0.0f) {
                p.y = 0.0f;
                p.vy = -p.vy * bounce;
            } else if (p.y > height) {
                p.y = height;
                p.vy = -p.vy * bounce;
            }

            particles[id] = p;
        }
    )CLC";

    cl_int err = CL_SUCCESS;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "OpenCL platform not found: " << clErrorString(err) << '\n';
        return false;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clGetPlatformIDs failed: " << clErrorString(err) << '\n';
        return false;
    }

    cl_device_id device = nullptr;
    cl_platform_id chosenPlatform = nullptr;
    for (auto p : platforms) {
        cl_uint numDevices = 0;
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
        if (err == CL_SUCCESS && numDevices > 0) {
            chosenPlatform = p;
            break;
        }
    }

    if (!device) {
        for (auto p : platforms) {
            cl_uint numDevices = 0;
            err = clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &device, &numDevices);
            if (err == CL_SUCCESS && numDevices > 0) {
                chosenPlatform = p;
                break;
            }
        }
    }

    if (!device || !chosenPlatform) {
        std::cerr << "No OpenCL device found.\n";
        return false;
    }

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(chosenPlatform),
        0
    };

    cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateContext failed: " << clErrorString(err) << '\n';
        return false;
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateCommandQueue failed: " << clErrorString(err) << '\n';
        clReleaseContext(context);
        return false;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateProgramWithSource failed: " << clErrorString(err) << '\n';
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "OpenCL build error:\n" << log << '\n';
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "step_particles", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateKernel failed: " << clErrorString(err) << '\n';
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    cl_mem buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        particles.size() * sizeof(Particle),
        particles.data(),
        &err
    );
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateBuffer failed: " << clErrorString(err) << '\n';
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &par.width);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &par.height);
    err |= clSetKernelArg(kernel, 3, sizeof(float), &par.gravityCenterX);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &par.gravityCenterY);
    err |= clSetKernelArg(kernel, 5, sizeof(float), &par.attraction);
    err |= clSetKernelArg(kernel, 6, sizeof(float), &par.damping);
    err |= clSetKernelArg(kernel, 7, sizeof(float), &par.bounce);
    err |= clSetKernelArg(kernel, 8, sizeof(float), &par.dt);
    if (err != CL_SUCCESS) {
        std::cerr << "clSetKernelArg failed: " << clErrorString(err) << '\n';
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    size_t globalSize = particles.size();
    for (int s = 0; s < par.steps; ++s) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "clEnqueueNDRangeKernel failed: " << clErrorString(err) << '\n';
            clReleaseMemObject(buffer);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            return false;
        }
        clFinish(queue);
    }

    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, particles.size() * sizeof(Particle), particles.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueReadBuffer failed: " << clErrorString(err) << '\n';
        clReleaseMemObject(buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return true;
}

static void writePPM(const std::string &path, const std::vector<Particle> &particles, int width, int height) {
    std::vector<uint8_t> image(width * height * 3, 0);
    for (const auto &p : particles) {
        int x = std::clamp(static_cast<int>(p.x), 0, width - 1);
        int y = std::clamp(static_cast<int>(p.y), 0, height - 1);
        size_t idx = static_cast<size_t>(y) * width * 3 + static_cast<size_t>(x) * 3;
        image[idx + 0] = 255;
        image[idx + 1] = 255;
        image[idx + 2] = 255;
    }

    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(image.data()), static_cast<std::streamsize>(image.size()));
}

static double runSequential(std::vector<Particle> particles, const Params &par) {
    double t0 = nowSeconds();
    for (int s = 0; s < par.steps; ++s) {
        stepSequential(particles, par);
        if (par.outputEvery > 0 && s % par.outputEvery == 0) {
            writePPM("output_seq_" + std::to_string(s) + ".ppm", particles, par.width, par.height);
        }
    }
    double t1 = nowSeconds();
    std::cout << "Sequential final sample: (" << particles[0].x << ", " << particles[0].y << ")\n";
    return t1 - t0;
}

static double runOpenMP(std::vector<Particle> particles, const Params &par) {
    double t0 = nowSeconds();
    for (int s = 0; s < par.steps; ++s) {
        stepOpenMP(particles, par);
        if (par.outputEvery > 0 && s % par.outputEvery == 0) {
            writePPM("output_omp_" + std::to_string(s) + ".ppm", particles, par.width, par.height);
        }
    }
    double t1 = nowSeconds();
    std::cout << "OpenMP final sample: (" << particles[0].x << ", " << particles[0].y << ")\n";
    return t1 - t0;
}

static double runOpenCL(std::vector<Particle> particles, const Params &par, bool &ok) {
    double t0 = nowSeconds();
    ok = stepOpenCL(particles, par);
    double t1 = nowSeconds();
    if (ok) {
        std::cout << "OpenCL final sample: (" << particles[0].x << ", " << particles[0].y << ")\n";
    }
    return t1 - t0;
}

int main(int argc, char **argv) {
    Params par;
    if (argc > 1) par.numParticles = std::max(1, std::atoi(argv[1]));
    if (argc > 2) par.steps = std::max(1, std::atoi(argv[2]));
    if (argc > 3) omp_set_num_threads(std::max(1, std::atoi(argv[3])));

    std::cout << "Particles: " << par.numParticles << '\n';
    std::cout << "Steps: " << par.steps << '\n';
    std::cout << "OpenMP threads: " << omp_get_max_threads() << '\n';

    auto base = makeParticles(par.numParticles, par.width, par.height);

    double tseq = runSequential(base, par);
    double tomp = runOpenMP(base, par);

    bool openclOk = false;
    double tocl = runOpenCL(base, par, openclOk);

    std::cout << "\n=== Benchmark ===\n";
    std::cout << "Sequential: " << tseq << " s\n";
    std::cout << "OpenMP    : " << tomp << " s\n";
    std::cout << "OpenCL    : " << (openclOk ? std::to_string(tocl) + " s" : std::string("not available")) << '\n';

    if (tomp > 0.0) {
        std::cout << "OpenMP speedup vs sequential: " << (tseq / tomp) << "x\n";
    }
    if (openclOk && tocl > 0.0) {
        std::cout << "OpenCL speedup vs sequential: " << (tseq / tocl) << "x\n";
    }

    std::cout << "\nGenerated sample images: output_seq_*.ppm and output_omp_*.ppm\n";
    return 0;
}

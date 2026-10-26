// Kernel tracer loaded into any CUDA process through CUDA_INJECTION64_PATH: records every kernel (graph nodes
// included) as "device,start_ns,end_ns,graph_node:stream,name" lines in $CUPTI_TRACE_OUT.<pid>.csv. A thread flushes CUPTI's
// buffers every second, so a process killed later still leaves what it ran.
//
//   CUDA_INJECTION64_PATH=/path/libcupti_trace.so CUPTI_TRACE_OUT=/path/trace ./app
#include <cupti.h>

#include <cxxabi.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

static std::FILE* g_out = nullptr;
static std::mutex g_mu;

static void CUPTIAPI buffer_requested(uint8_t** buf, size_t* size, size_t* max_records) {
    *size = 8 << 20;
    *buf = (uint8_t*) aligned_alloc(8, *size);
    *max_records = 0;
}

static void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
    CUpti_Activity* rec = nullptr;
    std::lock_guard<std::mutex> lock(g_mu);
    while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
        if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
        const auto* k = (const CUpti_ActivityKernel9*) rec;
        int st = 0;
        char* dem = abi::__cxa_demangle(k->name, nullptr, nullptr, &st);
        std::string name = st == 0 && dem ? dem : k->name;
        std::free(dem);
        for (size_t a; (a = name.find("(anonymous namespace)::")) != std::string::npos;) name.erase(a, 23);
        // drop the parameter list: the first '(' outside template arguments (those hold casts like "(ggml_type)8")
        int depth = 0;
        for (size_t i = 0; i < name.size(); ++i) {
            if (name[i] == '<') ++depth;
            else if (name[i] == '>') --depth;
            else if (name[i] == '(' && depth == 0) { name.resize(i); break; }
        }
        for (char& c : name)
            if (c == ',') c = ';';
        std::fprintf(g_out, "%u,%llu,%llu,%llu:%u,%s\n", k->deviceId, (unsigned long long) k->start,
                     (unsigned long long) k->end, (unsigned long long) k->graphNodeId, k->streamId, name.c_str());
    }
    std::fflush(g_out);
    std::free(buf);
}

extern "C" int InitializeInjection(void) {
    const char* base = std::getenv("CUPTI_TRACE_OUT");
    char path[512];
    std::snprintf(path, sizeof path, "%s.%d.csv", base ? base : "/tmp/cupti_trace", (int) getpid());
    g_out = std::fopen(path, "w");
    if (!g_out) return 0;
    std::fprintf(g_out, "device,start_ns,end_ns,graph_node:stream,name\n");
    if (cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed) != CUPTI_SUCCESS) return 0;
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS) return 0;
    std::atexit([] { cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED); });
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            cuptiActivityFlushAll(0);
        }
    }).detach();
    return 1;
}

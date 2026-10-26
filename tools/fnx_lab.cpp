// fnx lab: the model loaded once, fnx_run's commands run against it from a shared library that a rebuild can replace.
// Commands come one a line from a named pipe (each writer may open, write and close it), the output goes to stdout:
//   <fnx_run mode and arguments>   e.g. "read 30720" (the model's path is put in)
//   set VAR=VALUE / unset VAR       an engine setting for the commands after (the library is reloaded before the
//                                   next one, so settings read once per process are read again); FNX_ARENA,
//                                   FNX_EXPERT_RAM_GIB and FNX_EXPERT_PROFILE place the routed experts again (attached)
//   reload                          the library again from disk (after `make build/libfnx_lab.so`)
//   quit
// Each command ends with "== done <exit code> in <seconds> s: <command>". Attached to a holder (bin/fnx_hold, which
// keeps the weights in shared memory and runs no kernel) a worker that faults is replaced in seconds by tools/lab.sh; a
// change to the loader (model.cpp, model.hpp, gguf.*, store.*) needs a new holder; everything the engine is (kernels,
// engine, prompt path, session, fnx_run's modes) is in the library.
//   fnx_lab --attach SOCKET FIFO LIB.so       a worker of the holder at SOCKET (tools/lab.sh's way)
//   fnx_lab MODEL.gguf FIFO LIB.so            the model loaded by this process alone
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/model.hpp"
#include "../src/store.hpp"
#include "lab_proto.hpp"

using RunCmd = int (*)(fnx::Model*, int, char**);

struct Lib {
    std::string path;
    void* h = nullptr;
    RunCmd run = nullptr;
    void load() {
        h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error(std::string("fnx_lab: ") + dlerror());
        run = reinterpret_cast<RunCmd>(dlsym(h, "fnx_run_cmd"));
        if (!run) throw std::runtime_error(std::string("fnx_lab: ") + path + " has no fnx_run_cmd: " + dlerror());
    }
    void unload() {
        if (h && dlclose(h) != 0) throw std::runtime_error(std::string("fnx_lab: dlclose: ") + dlerror());
        h = nullptr;
        run = nullptr;
    }
};

// the settings Model::place_experts reads: a change places the experts again (an attached worker only)
static bool expert_setting(const std::string& var) {
    return var == "FNX_ARENA" || var == "FNX_EXPERT_RAM_GIB" || var == "FNX_EXPERT_PROFILE";
}

int main(int argc, char** argv) {
    const bool attach = argc == 5 && std::string(argv[1]) == "--attach";
    if (argc != 4 && !attach) {
        std::fprintf(stderr, "usage: fnx_lab MODEL.gguf FIFO LIB.so | fnx_lab --attach SOCKET FIFO LIB.so\n");
        return 2;
    }
    const char* fifo_path = argv[argc - 2];
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);
    const auto t_load = std::chrono::steady_clock::now();
    fnx::Store store;
    std::unique_ptr<fnx::Model> mp;
    std::string model_path;
    if (attach) {
        std::string image;
        std::vector<fnx::Store::Chunk> chunks;
        lab::fetch(argv[2], model_path, chunks, image);
        store.attach(chunks);
        for (const auto& c : chunks) ::close(c.fd);
        mp = std::make_unique<fnx::Model>(model_path, fnx::Load::ATTACH, &store, &image);
    } else {
        model_path = argv[1];
        mp = std::make_unique<fnx::Model>(model_path);
    }
    fnx::Model& m = *mp;
    Lib lib{argv[argc - 1]};
    lib.load();
    std::printf("== ready: model %s in %.1f s\n", attach ? "attached" : "loaded",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());
    bool stale = false;     // a setting changed since the library was loaded
    bool replace = false;   // an expert setting changed since the experts were placed
    for (;;) {
        std::ifstream fifo(fifo_path);   // blocks until a writer opens it; its end-of-file is that writer closing it
        if (!fifo) {
            std::fprintf(stderr, "fnx_lab: cannot open %s\n", fifo_path);
            return 1;
        }
        std::string line;
        while (std::getline(fifo, line)) {
            if (line.empty()) continue;
            const auto t0 = std::chrono::steady_clock::now();
            int rc = 0;
            try {
                if (line == "quit") {
                    std::printf("== done 0 in 0.0 s: quit\n");
                    return 0;
                } else if (line == "reload") {
                    lib.unload();
                    lib.load();
                    stale = false;
                } else if (line.rfind("set ", 0) == 0 || line.rfind("unset ", 0) == 0) {
                    const bool set = line[0] == 's';
                    const std::string kv = line.substr(set ? 4 : 6);
                    const size_t eq = kv.find('=');
                    // an empty value would still count as set (the engine checks presence): turning off is unset
                    if (set && (eq == std::string::npos || eq == 0 || eq + 1 == kv.size()))
                        throw std::runtime_error("set needs VAR=VALUE (to turn a setting off: unset VAR)");
                    const std::string var = set ? kv.substr(0, eq) : kv;
                    if (expert_setting(var) && !attach) throw std::runtime_error(var + " is the loader's: only an attached worker places the experts again");
                    if (set) setenv(var.c_str(), kv.substr(eq + 1).c_str(), 1);
                    else unsetenv(var.c_str());
                    stale = true;
                    replace = replace || expert_setting(var);
                } else {
                    if (stale) {
                        lib.unload();
                        lib.load();
                        stale = false;
                    }
                    if (replace) {
                        const auto tp = std::chrono::steady_clock::now();
                        m.free_experts();
                        m.ram_extra = 0;   // new settings: an engine finds VRAM short again if it still is
                        m.place_experts();
                        replace = false;
                        std::printf("fnx_lab: experts placed again in %.1f s\n",
                                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
                    }
                    std::vector<std::string> words{"fnx_run", model_path};
                    std::istringstream is(line);
                    for (std::string w; is >> w;) words.push_back(w);
                    std::vector<char*> args;
                    for (std::string& w : words) args.push_back(w.data());
                    args.push_back(nullptr);
                    rc = lib.run(&m, (int) words.size(), args.data());
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "fnx_lab: %s\n", e.what());
                rc = 1;
            }
            std::fflush(stderr);
            // a fault in a kernel (an illegal address, a trap) leaves this process's contexts unusable: say so and end,
            // so tools/lab.sh attaches a new worker (the holder keeps the model)
            const char* dead = nullptr;
            for (int c = 0; c < fnx::NCARD && !dead; ++c)
                if (cudaSetDevice(c) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) dead = cudaGetErrorString(cudaGetLastError());
            std::printf("== done %d in %.1f s: %s\n", dead ? 70 : rc,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), line.c_str());
            if (dead) {
                std::printf("== this worker's CUDA contexts are broken (%s): it ends\n", dead);
                return 70;
            }
        }
    }
}

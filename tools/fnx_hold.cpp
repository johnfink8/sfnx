// The lab's holder: loads the model once into shared memory (src/store.hpp) and hands it to workers (bin/fnx_lab
// --attach) on a unix socket. It runs no kernel, so nothing a worker does can take the model down; a worker that dies
// is replaced in seconds (tools/lab.sh).
//   fnx_hold MODEL.gguf SOCKET          prints "== holding" once loaded; a client sending "QUIT" ends it
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "../src/model.hpp"
#include "../src/store.hpp"
#include "lab_proto.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: fnx_hold MODEL.gguf SOCKET\n");
        return 2;
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        const auto t0 = std::chrono::steady_clock::now();
        fnx::Store store;
        fnx::Model m(argv[1], fnx::Load::HOLD, &store);
        const std::string image = m.image();
        double dev = 0, host = 0;
        for (int c = 0; c < fnx::NCARD; ++c) dev += store.bytes(c), host += store.bytes(fnx::NCARD + c);
        ::unlink(argv[2]);
        const int ls = ::socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un a{};
        a.sun_family = AF_UNIX;
        std::snprintf(a.sun_path, sizeof a.sun_path, "%s", argv[2]);
        if (ls < 0 || ::bind(ls, (sockaddr*) &a, sizeof a) != 0 || ::listen(ls, 4) != 0) throw std::runtime_error(std::string("cannot listen on ") + argv[2]);
        std::printf("== holding: %s in %.1f s (%.1f GiB on the cards, %.1f GiB of expert copies in host memory, %zu chunks)\n", argv[1],
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), dev / (1ull << 30), host / (1ull << 30),
                    store.chunks().size());
        for (;;) {
            const int s = ::accept(ls, nullptr, nullptr);
            if (s < 0) continue;
            try {
                char cmd[4];
                lab::recv_all(s, cmd, 4);
                if (!std::memcmp(cmd, "QUIT", 4)) {
                    std::printf("== quit\n");
                    ::close(s);
                    return 0;
                }
                if (std::memcmp(cmd, "ATC2", 4)) throw std::runtime_error("unknown command (a worker from before the stamped wire?)");
                if (lab::serve(s, argv[1], store.chunks(), image)) std::printf("fnx_hold: a worker attached\n");
                else std::printf("fnx_hold: refused a worker built from other loader sources\n");
            } catch (const std::exception& e) {
                std::fprintf(stderr, "fnx_hold: %s\n", e.what());
            }
            ::close(s);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fnx_hold: %s\n", e.what());
        return 1;
    }
}

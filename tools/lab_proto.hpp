// The lab's holder (fnx_hold) and its workers (fnx_lab --attach) over a unix socket: a worker sends "ATC2" and its build stamp, and gets the
// model's path, the Store's chunks (their file descriptors by SCM_RIGHTS) and the cards' tables (Model::image).
#pragma once

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/store.hpp"

namespace lab {

inline void send_all(int s, const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n) {
        const ssize_t k = ::send(s, c, n, MSG_NOSIGNAL);
        if (k <= 0) throw std::runtime_error("lab: send failed");
        c += k, n -= (size_t) k;
    }
}
inline void recv_all(int s, void* p, size_t n) {
    char* c = (char*) p;
    while (n) {
        const ssize_t k = ::recv(s, c, n, 0);
        if (k <= 0) throw std::runtime_error("lab: the holder closed the connection");
        c += k, n -= (size_t) k;
    }
}
inline void send_str(int s, const std::string& v) {
    const uint64_t n = v.size();
    send_all(s, &n, 8);
    send_all(s, v.data(), v.size());
}
inline std::string recv_str(int s) {
    uint64_t n = 0;
    recv_all(s, &n, 8);
    std::string v(n, '\0');
    recv_all(s, v.data(), n);
    return v;
}
constexpr int FD_BATCH = 64;
inline void send_fds(int s, const int* fds, int n) {
    char ctl[CMSG_SPACE(sizeof(int) * FD_BATCH)] = {};
    char b = 'f';
    iovec io{&b, 1};
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = ctl;
    m.msg_controllen = CMSG_SPACE(sizeof(int) * n);
    cmsghdr* c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int) * n);
    std::memcpy(CMSG_DATA(c), fds, sizeof(int) * n);
    if (::sendmsg(s, &m, MSG_NOSIGNAL) != 1) throw std::runtime_error("lab: sending descriptors failed");
}
inline void recv_fds(int s, int* fds, int n) {
    char ctl[CMSG_SPACE(sizeof(int) * FD_BATCH)] = {};
    char b;
    iovec io{&b, 1};
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = ctl;
    m.msg_controllen = sizeof ctl;
    if (::recvmsg(s, &m, 0) != 1) throw std::runtime_error("lab: receiving descriptors failed");
    cmsghdr* c = CMSG_FIRSTHDR(&m);
    if (!c || c->cmsg_type != SCM_RIGHTS || c->cmsg_len != CMSG_LEN(sizeof(int) * n)) throw std::runtime_error("lab: a short descriptor batch");
    std::memcpy(fds, CMSG_DATA(c), sizeof(int) * n);
}

// Holder and worker are separate binaries, and the worker is rebuilt while the holder keeps running: one built from
// other loader sources would read the held weights at the wrong places with no error. The worker sends this build's
// stamp (the loader sources' hash, FNX_LOADER_REV from the Makefile, and the wire's version) and the holder refuses
// a different one, so a loader change asks for `unload` + `start` instead of giving wrong logits.
#ifndef FNX_LOADER_REV
#error "FNX_LOADER_REV: the Makefile passes the loader sources' hash"
#endif
constexpr uint32_t WIRE_VERSION = 2;
inline std::string build_stamp() { return "fnx-lab wire " + std::to_string(WIRE_VERSION) + " loader " + FNX_LOADER_REV; }

// the holder's side of one "ATC2" (after the command word): the worker's stamp, then "" and the model, or why not
inline bool serve(int s, const std::string& model, const std::vector<fnx::Store::Chunk>& chunks, const std::string& image) {
    const std::string theirs = recv_str(s), ours = build_stamp();
    if (theirs != ours) {
        send_str(s, "the holder was built from other loader sources (holder: " + ours + "; worker: " + theirs +
                        "): tools/lab.sh unload, then start");
        return false;
    }
    send_str(s, "");
    send_str(s, model);
    const uint64_t n = chunks.size();
    send_all(s, &n, 8);
    for (const auto& c : chunks) {   // fixed-width fields, not the struct's image (its padding is the compiler's)
        const int32_t loc = c.loc;
        const uint64_t va = c.va, size = c.size;
        send_all(s, &loc, 4);
        send_all(s, &va, 8);
        send_all(s, &size, 8);
    }
    std::vector<int> fds;
    for (const auto& c : chunks) fds.push_back(c.fd);
    for (size_t i = 0; i < fds.size(); i += FD_BATCH) send_fds(s, fds.data() + i, (int) std::min<size_t>(FD_BATCH, fds.size() - i));
    send_str(s, image);
    return true;
}
// a worker's: connect to the holder's socket and take everything (the chunks' fd fields are this process's descriptors)
inline void fetch(const std::string& path, std::string& model, std::vector<fnx::Store::Chunk>& chunks, std::string& image) {
    const int s = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    if (path.size() >= sizeof a.sun_path) throw std::runtime_error("lab: socket path too long");
    std::strcpy(a.sun_path, path.c_str());
    if (s < 0 || ::connect(s, (sockaddr*) &a, sizeof a) != 0) throw std::runtime_error("lab: no holder at " + path + " (tools/lab.sh start)");
    send_all(s, "ATC2", 4);
    send_str(s, build_stamp());
    std::string refused;
    try {
        refused = recv_str(s);
    } catch (const std::exception&) {
        throw std::runtime_error("lab: the holder closed the connection at the handshake (a holder from before the stamped wire?): "
                                 "tools/lab.sh unload, then start");
    }
    if (!refused.empty()) throw std::runtime_error("lab: " + refused);
    model = recv_str(s);
    uint64_t n = 0;
    recv_all(s, &n, 8);
    chunks.resize(n);
    for (auto& c : chunks) {
        int32_t loc = 0;
        uint64_t va = 0, size = 0;
        recv_all(s, &loc, 4);
        recv_all(s, &va, 8);
        recv_all(s, &size, 8);
        c.loc = loc, c.va = va, c.size = size;
    }
    std::vector<int> fds(n);
    for (size_t i = 0; i < n; i += FD_BATCH) recv_fds(s, fds.data() + i, (int) std::min<size_t>(FD_BATCH, n - i));
    for (size_t i = 0; i < n; ++i) chunks[i].fd = fds[i];
    image = recv_str(s);
    ::close(s);
}

}  // namespace lab

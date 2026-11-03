#include "ple_io.hpp"
#include "env.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

namespace fnx {
namespace {

constexpr uint64_t PAGE = 4096;
constexpr size_t SLAB_UNITS = 4096;   // reads in flight per round: 32 MiB of slab at two pages a unit

uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// env_int within [lo, hi]
uint64_t env_in(const char* name, long long def, long long lo, long long hi) {
    const long long v = env_int(name, def);
    if (v < lo || v > hi) throw std::runtime_error(std::string(name) + ": " + std::to_string(v) + ", not " + std::to_string(lo) + " to " + std::to_string(hi));
    return (uint64_t) v;
}

// "0" / "1" from the sysfs of the block device holding fd's file (the partition's disk when it is a partition);
// empty when the file system has no block device sysfs knows (NFS, overlay, ZFS)
std::string rotational_of(int fd) {
    struct stat st;
    if (fstat(fd, &st) != 0) return "";
    const std::string dev = "/sys/dev/block/" + std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev));
    for (const char* q : {"/queue/rotational", "/../queue/rotational"}) {
        std::ifstream f(dev + q);
        std::string v;
        if (f >> v) return v;
    }
    return "";
}

}  // namespace

PleRows::Mode PleRows::mode_from_env() {
    const char* s = std::getenv("FNX_NGRAM");
    Mode m;
    if (!s || !std::strcmp(s, "ram")) m = Mode::RAM;
    else if (!std::strcmp(s, "disk")) m = Mode::DISK;
    else throw std::runtime_error(std::string("FNX_NGRAM: '") + s + "', not ram or disk");
    // the disk's settings are checked whatever the mode: one set beside FNX_NGRAM=ram would change nothing, silently
    env_in("FNX_NGRAM_CACHE_MIB", 256, 0, 1 << 20);
    env_in("FNX_NGRAM_THREADS", 32, 1, 256);
    for (const char* v : {"FNX_NGRAM_CACHE_MIB", "FNX_NGRAM_THREADS"})
        if (m == Mode::RAM && std::getenv(v)) throw std::runtime_error(std::string(v) + " is set, and only FNX_NGRAM=disk reads it");
    return m;
}

PleRows::PleRows(const Gguf& g, const Tensor& t, Mode mode) : mode_(mode), t_(t), rb_(t.row_bytes()) {
    if (rb_ == 0 || rb_ > PAGE) throw std::runtime_error("n-gram table: " + t.name + "'s rows are " + std::to_string(rb_) + " bytes, not 1 to 4096");
    if (mode_ == Mode::RAM) return;
    if (t.shard < 0) throw std::runtime_error("n-gram table: " + t.name + " is not in a file, so it cannot be read from disk");
    path_ = g.shard_path(t.shard);
    file_off_ = t.offset;
    fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECT);
    direct_ = fd_ >= 0;
    if (fd_ < 0) {
        if (errno != EINVAL) throw std::runtime_error("n-gram table: cannot open " + path_ + ": " + std::strerror(errno));
        // a file system without O_DIRECT (tmpfs): its pages are in RAM already, so the page cache costs nothing more
        fd_ = ::open(path_.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("n-gram table: cannot open " + path_ + ": " + std::strerror(errno));
        posix_fadvise(fd_, (off_t) file_off_, (off_t) t.nbytes, POSIX_FADV_RANDOM);
        std::fprintf(stderr, "n-gram table: %s refuses O_DIRECT: rows read through the page cache\n", path_.c_str());
    }
    // a spinning disk serves ~100-200 random reads a second, against ~30,000 a 4k-token prompt chunk needs. A file
    // system with no block device in sysfs (NFS, overlay, ZFS, btrfs) cannot be told: describe() says so, and the probe's
    // rate at start is the measure
    rot_ = rotational_of(fd_);
    if (rot_ == "1") {
        ::close(fd_);
        throw std::runtime_error("n-gram table: " + path_ + " is on a rotational disk, too slow for FNX_NGRAM=disk (a random read per row): "
                                 "move the model to an SSD or use FNX_NGRAM=ram");
    }
    const uint64_t cache_mib = env_in("FNX_NGRAM_CACHE_MIB", 256, 0, 1 << 20);
    threads_ = (int) env_in("FNX_NGRAM_THREADS", 32, 1, 256);
    sets_ = (cache_mib << 20) / rb_ / WAYS;
    tag_.assign(sets_ * WAYS, 0);
    ref_.assign(sets_ * WAYS, 0);
    hand_.assign(sets_, 0);
    data_.resize(sets_ * WAYS * rb_);
    slab_ = (uint8_t*) std::aligned_alloc(PAGE, SLAB_UNITS * 2 * PAGE);
    if (!slab_) throw std::runtime_error("n-gram table: no memory for the read slab");
    lat_ns_.resize(SLAB_UNITS);
    go_ = std::make_unique<std::condition_variable[]>((size_t) threads_);
    for (int i = 0; i < threads_; ++i) pool_.emplace_back([this, i] { worker(i); });
}

PleRows::~PleRows() {
    for (std::future<void>& f : pending_) f.wait();   // they read through the pool
    {
        std::lock_guard<std::mutex> l(m_);
        stop_ = true;
    }
    for (int i = 0; i < threads_; ++i) go_[i].notify_all();
    for (std::thread& th : pool_) th.join();
    std::free(slab_);
    if (fd_ >= 0) ::close(fd_);
}

std::string PleRows::describe() const {
    if (mode_ == Mode::RAM) return "n-gram table: " + t_.name + " locked in RAM (FNX_NGRAM=ram)";
    return "n-gram table: " + t_.name + " read from " + path_ + (direct_ ? " (O_DIRECT" : " (page cache") +
           (rot_ == "0" ? ", an SSD)" : rot_.empty() ? ", its disk's type unknown: no sysfs entry for the file's device)" : ")") + ", " +
           std::to_string(threads_) +
           " threads, a row cache of " + std::to_string(sets_ * WAYS) + " rows (" + std::to_string((sets_ * WAYS * rb_) >> 20) + " MiB)";
}

void PleRows::worker(int w) {
    uint64_t seen = 0;
    for (;;) {
        const std::vector<Unit>* job;
        {
            std::unique_lock<std::mutex> l(m_);
            go_[w].wait(l, [&] { return stop_ || (gen_ != seen && w < active_); });
            if (stop_) return;
            seen = gen_;
            job = job_;
        }
        std::string err;
        for (size_t i; (i = next_.fetch_add(1)) < job->size();) {
            const Unit& u = (*job)[i];
            uint8_t* to = slab_ + i * 2 * PAGE;
            const size_t want = u.pages * PAGE;
            const off_t at = (off_t) (u.page * PAGE);
            const uint64_t t0 = now_ns();
            size_t got = 0;
            while (got < want && err.empty()) {
                const ssize_t r = ::pread(fd_, to + got, want - got, at + (off_t) got);
                if (r < 0 && errno == EINTR) continue;
                if (r < 0) err = "n-gram table: reading " + path_ + " at " + std::to_string(at + got) + ": " + std::strerror(errno);
                else if (r == 0) break;   // the file's end: the rows' bytes are checked against what came
                else {
                    got += (size_t) r;
                    // O_DIRECT reads come back short only at the file's end, and asking again from an unaligned offset
                    // is EINVAL on some file systems
                    if (direct_ && got < want) break;
                }
            }
            lat_ns_[i] = now_ns() - t0;
            // the rows this unit serves end within the table, which ends within the file; the bytes up to the table's end
            // must all have come
            const uint64_t need = std::min<uint64_t>(want, file_off_ + t_.nbytes - u.page * PAGE);
            if (err.empty() && got < need)
                err = "n-gram table: reading " + path_ + " at " + std::to_string(at) + ": " + std::to_string(got) + " of " + std::to_string(need) +
                      " bytes (the file is shorter than its table)";
        }
        // every worker of the round checks in, so none still reads the job when the next round rebuilds it
        std::lock_guard<std::mutex> l(m_);
        if (!err.empty() && err_.empty()) err_ = err;
        if (++done_ == (size_t) active_) finished_.notify_all();
    }
}

void PleRows::read_units(const std::vector<Unit>& units) {
    if (units.size() > SLAB_UNITS) throw std::runtime_error("n-gram table: a round of more reads than the slab holds");
    const uint64_t t0 = now_ns();
    {
        std::lock_guard<std::mutex> l(m_);
        job_ = &units;
        next_ = 0;
        done_ = 0;
        err_.clear();
        // as many workers as reads: waking all 32 for decode's handful cost prose2 ~2.5% (132.4 tok/s at 32 threads,
        // 135.7 at 8)
        active_ = (int) std::min<size_t>((size_t) threads_, units.size());
        ++gen_;
    }
    for (int i = 0; i < active_; ++i) go_[i].notify_one();
    std::unique_lock<std::mutex> l(m_);
    finished_.wait(l, [&] { return done_ == (size_t) active_; });
    reads_ += units.size();
    read_ns_ += now_ns() - t0;
    if (!err_.empty()) throw std::runtime_error(err_);
}

bool PleRows::lookup(uint64_t row, uint8_t* dst) {
    if (!sets_) return false;
    const uint64_t s = row % sets_;
    for (int w = 0; w < WAYS; ++w) {
        const size_t slot = s * WAYS + (size_t) w;
        if (tag_[slot] == row + 1) {
            ref_[slot] = 1;
            std::memcpy(dst, &data_[slot * rb_], rb_);
            return true;
        }
    }
    return false;
}

void PleRows::insert(uint64_t row, const uint8_t* src) {
    if (!sets_) return;
    const uint64_t s = row % sets_;
    uint8_t& h = hand_[s];
    for (;;) {
        const size_t slot = s * WAYS + h;
        h = (uint8_t) ((h + 1) % WAYS);
        if (tag_[slot] && ref_[slot]) {
            ref_[slot] = 0;
            continue;
        }
        tag_[slot] = row + 1;
        ref_[slot] = 0;
        std::memcpy(&data_[slot * rb_], src, rb_);
        return;
    }
}

void PleRows::prefetch(std::vector<uint64_t> rows) {
    if (mode_ != Mode::DISK || rows.empty()) return;
    std::lock_guard<std::mutex> l(pending_m_);
    pending_.push_back(std::async(std::launch::async, [this, rows = std::move(rows)] {
        std::vector<uint8_t> scratch(rows.size() * rb_);
        gather_now(rows.data(), rows.size(), scratch.data());
    }));
}

void PleRows::gather(const uint64_t* rows, size_t n, uint8_t* dst) {
    std::vector<std::future<void>> before;
    {
        std::lock_guard<std::mutex> l(pending_m_);
        before.swap(pending_);
    }
    // every one waited for before the first failure is thrown: none outlives this call reading through the pool
    std::exception_ptr err;
    for (std::future<void>& f : before) try {
            f.get();
        } catch (...) {
            if (!err) err = std::current_exception();
        }
    if (err) std::rethrow_exception(err);
    gather_now(rows, n, dst);
}

void PleRows::gather_now(const uint64_t* rows, size_t n, uint8_t* dst) {
    std::lock_guard<std::mutex> gl(gather_m_);
    const uint64_t nrows = (uint64_t) t_.nrows();
    for (size_t i = 0; i < n; ++i)
        if (rows[i] >= nrows) throw std::runtime_error("n-gram table: row " + std::to_string(rows[i]) + " past " + t_.name + "'s " + std::to_string(nrows));
    lookups_ += n;
    if (mode_ == Mode::RAM) {
        for (size_t i = 0; i < n; ++i) std::memcpy(dst + i * rb_, t_.data + rows[i] * rb_, rb_);
        hits_ += n;
        return;
    }
    // the lookups the cache misses, by row, so a row asked for twice is read once
    std::vector<std::pair<uint64_t, size_t>> miss;
    for (size_t i = 0; i < n; ++i)
        if (!lookup(rows[i], dst + i * rb_)) miss.emplace_back(rows[i], i);
    hits_ += n - miss.size();
    std::sort(miss.begin(), miss.end());
    std::vector<Unit> units;
    std::vector<std::pair<uint64_t, size_t>> round;   // this round's misses
    size_t k = 0;
    while (k < miss.size()) {
        units.clear();
        round.clear();
        // distinct rows in order: their pages ascend, so a unit is shared only with the unit just before
        while (k < miss.size() && units.size() < SLAB_UNITS) {
            const uint64_t row = miss[k].first, b = file_off_ + row * rb_;
            const uint64_t p0 = b / PAGE, p1 = (b + rb_ - 1) / PAGE;
            if (!units.empty() && units.back().page == p0) {
                units.back().pages = std::max<uint32_t>(units.back().pages, (uint32_t) (p1 - p0 + 1));
            } else if (!units.empty() && units.back().pages == 2 && units.back().page + 1 == p0 && p1 == p0) {
                // inside the second page of the unit before
            } else {
                units.push_back(Unit{p0, (uint32_t) (p1 - p0 + 1)});
            }
            for (; k < miss.size() && miss[k].first == row; ++k) round.push_back(miss[k]);
        }
        read_units(units);
        // each miss's bytes from the unit holding its first byte (units ascend by page)
        size_t u = 0;
        uint64_t last = UINT64_MAX;
        for (const auto& [row, i] : round) {
            const uint64_t b = file_off_ + row * rb_;
            while (u + 1 < units.size() && units[u + 1].page * PAGE <= b) ++u;
            const uint8_t* src = slab_ + u * 2 * PAGE + (b - units[u].page * PAGE);
            if (b + rb_ > (units[u].page + units[u].pages) * PAGE) throw std::runtime_error("n-gram table: a row past its read (planner)");
            std::memcpy(dst + i * rb_, src, rb_);
            if (row != last) insert(row, src);
            last = row;
        }
    }
}

PleRows::Stats PleRows::stats() const {
    return Stats{lookups_.load(), hits_.load(), reads_.load(), read_ns_.load()};
}

std::string PleRows::probe(int pages) {
    if (mode_ != Mode::DISK) throw std::runtime_error("n-gram table: the probe reads the disk (FNX_NGRAM=disk)");
    const uint64_t first = file_off_ / PAGE, last = (file_off_ + t_.nbytes - 1) / PAGE;
    std::mt19937_64 rng(0x9e3779b97f4a7c15ull);
    std::vector<uint64_t> ps;
    for (int i = 0; i < pages; ++i) ps.push_back(first + rng() % (last - first + 1));
    ps.push_back(last);   // the table's last page, which may be the file's: its short read is checked at start, not at a request
    std::sort(ps.begin(), ps.end());
    ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
    std::vector<uint64_t> lat;
    const uint64_t t0 = now_ns();
    for (size_t a = 0; a < ps.size(); a += SLAB_UNITS) {
        std::vector<Unit> units;
        for (size_t i = a; i < std::min(ps.size(), a + SLAB_UNITS); ++i) units.push_back(Unit{ps[i], 1});
        std::lock_guard<std::mutex> gl(gather_m_);
        read_units(units);
        lat.insert(lat.end(), lat_ns_.begin(), lat_ns_.begin() + (long) units.size());
    }
    const double s = (now_ns() - t0) / 1e9;
    std::sort(lat.begin(), lat.end());
    char buf[160];
    std::snprintf(buf, sizeof buf, "%.0f reads/s (%zu cold pages at %d threads), p50 %.0f us, p99 %.0f us", ps.size() / s, ps.size(), threads_,
                  lat[lat.size() / 2] / 1e3, lat[lat.size() * 99 / 100] / 1e3);
    return buf;
}

}  // namespace fnx

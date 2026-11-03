#pragma once
// Rows of the PLE table (per_layer_token_embd: 28.8 GB in IQ4_NL, 54.4 GB in Q8_0) for the host's gather.
// FNX_NGRAM=ram (the default): the mapping, locked in RAM by the loader. FNX_NGRAM=disk: nothing of the table in RAM but a
// row cache; rows not in it are read with O_DIRECT from the file, one page each (two when a row crosses a page), by a
// pool of threads doing blocking preads (queue depth = threads). A new prompt is ~45% rows it has not used before
// (ref/long2, long64, long164: 41.8-49.4% distinct), so the cache pays across requests that repeat text, not within one.
// The page cache is no use here: a row is 90 or 170 bytes of a 4 KiB page, and rows land on pages at random.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gguf.hpp"

namespace fnx {

// The table rows of positions p0 .. p0 + T - 1: for each, the n-grams of its token and the mult.size() - 1 before it
// (sequence start and eos cut the window), each order's mixed hash into heads/(orders - 1) heads, head h at
// mixed % vocab[h] + off[h]. rows[t * off.size() + h]. tok_at(q): the token at position q >= 0, as the n-grams read it
struct PleHash {
    std::vector<uint64_t> mult, off, vocab;
    int64_t eos = 0;
};
template <class TokAt>
void ple_window_rows(const PleHash& ph, int p0, int T, const TokAt& tok_at, uint64_t* rows) {
    const int orders = (int) ph.mult.size(), heads = (int) ph.off.size(), per = heads / (orders - 1);
    std::vector<int64_t> ctx((size_t) orders);
    for (int t = 0; t < T; ++t) {
        const int p = p0 + t;
        ctx[0] = tok_at(p);
        bool cut = false;
        for (int s = 1; s < orders; ++s) {
            const int q = p - s;
            const int64_t tk = q >= 0 ? (int64_t) tok_at(q) : -1;
            cut = cut || tk < 0 || tk == ph.eos;
            ctx[(size_t) s] = cut ? ph.eos : tk;
        }
        for (int n = 2; n <= orders; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * ph.mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[(size_t) j] * ph.mult[(size_t) j];
            for (int gi = 0; gi < per; ++gi) {
                const int h = (n - 2) * per + gi;
                rows[(size_t) t * heads + h] = mixed % ph.vocab[(size_t) h] + ph.off[(size_t) h];
            }
        }
    }
}

class PleRows {
public:
    enum class Mode { RAM, DISK };
    struct Stats {
        uint64_t lookups = 0, hits = 0, reads = 0, read_ns = 0;
    };
    // FNX_NGRAM, FNX_NGRAM_CACHE_MIB, FNX_NGRAM_THREADS: read here, refused when malformed
    static Mode mode_from_env();
    PleRows(const Gguf& g, const Tensor& t, Mode mode);
    ~PleRows();
    PleRows(const PleRows&) = delete;
    PleRows& operator=(const PleRows&) = delete;

    Mode mode() const { return mode_; }
    size_t row_bytes() const { return rb_; }
    // the bytes of rows[0..n) to dst (n * row_bytes()); throws on a failed read. One caller at a time (a lock). It
    // first waits for the prefetches before it, and throws the first one's failure
    void gather(const uint64_t* rows, size_t n, uint8_t* dst);
    // DISK: the rows read into the cache on a thread of their own, for a gather to come (decode: the next window's
    // tokens as they become known, while the draft layer runs); RAM: nothing
    void prefetch(std::vector<uint64_t> rows);
    Stats stats() const;
    // DISK: time `pages` random reads of the table's pages, cold (O_DIRECT): "N reads/s, p50 X us, p99 Y us"
    std::string probe(int pages);
    std::string describe() const;   // the mode and its settings, for the start-up line

private:
    struct Unit {
        uint64_t page;   // the table file's page the read starts at
        uint32_t pages;  // 1, or 2 for a row that crosses into the next
    };
    void read_units(const std::vector<Unit>& units);   // into slab_, unit i at i * 2 pages; throws on failure
    void worker(int w);
    bool lookup(uint64_t row, uint8_t* dst);
    void insert(uint64_t row, const uint8_t* src);

    Mode mode_;
    const Tensor& t_;
    size_t rb_ = 0;
    uint64_t file_off_ = 0;   // the table's first byte in its shard
    std::string path_;
    int fd_ = -1;
    bool direct_ = false;
    std::string rot_;   // sysfs's rotational for the file's disk: "0", "1", or "" when it has none

    // the row cache: 8-way set-associative, CLOCK within a set
    static constexpr int WAYS = 8;
    uint64_t sets_ = 0;
    std::vector<uint64_t> tag_;       // [sets * WAYS]: row + 1, 0 empty
    std::vector<uint8_t> ref_;        // [sets * WAYS]
    std::vector<uint8_t> hand_;       // [sets]
    std::vector<uint8_t> data_;       // [sets * WAYS * rb]

    // the read pool
    int threads_ = 0;
    uint8_t* slab_ = nullptr;   // SLAB_UNITS * 2 pages, page-aligned
    const std::vector<Unit>* job_ = nullptr;
    std::vector<uint64_t> lat_ns_;   // per unit of the current job
    std::atomic<size_t> next_{0};
    size_t done_ = 0;   // workers through the current round
    uint64_t gen_ = 0;
    bool stop_ = false;
    std::string err_;
    std::mutex m_;
    int active_ = 0;   // the workers of the current round: 0 .. active_ - 1
    std::unique_ptr<std::condition_variable[]> go_;   // a worker's own
    std::condition_variable finished_;
    std::vector<std::thread> pool_;

    void gather_now(const uint64_t* rows, size_t n, uint8_t* dst);   // under gather_m_
    std::mutex gather_m_;
    // the prefetch thread (DISK): the newest of the queued requests (each a later one's prefix, decode's window as it
    // grows), its failure kept for the next gather
    void prefetcher();
    std::mutex pf_m_;
    std::condition_variable pf_go_, pf_idle_;
    std::vector<uint64_t> pf_next_;
    bool pf_queued_ = false, pf_busy_ = false, pf_stop_ = false;
    std::exception_ptr pf_err_;
    std::thread pf_thread_;
    std::atomic<uint64_t> lookups_{0}, hits_{0}, reads_{0}, read_ns_{0};
};

}  // namespace fnx

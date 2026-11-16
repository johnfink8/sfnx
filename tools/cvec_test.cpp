// Host-only test of the control vector loader (src/cvec.cpp): the settings parse, and a direction file is checked
// against the model it will be applied to - the shapes, the names, the unit norm, the hints, the ranges. Writes the
// files it needs, one good and several that must be refused with a message naming what is wrong; when the real
// direction file the feature is built for is given, loads that too.
//   make bin/cvec_test && bin/cvec_test [DIRECTION.gguf]
#include "../src/cvec.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace fnx;

namespace {

int fails = 0;
void ok(bool what, const char* what_s) {
    if (what) return;
    std::printf("FAIL: %s\n", what_s);
    ++fails;
}
void named(const std::string& err, const std::string& needle, const char* what_s) {
    if (err.find(needle) != std::string::npos) return;
    std::printf("FAIL: %s: the error names nothing useful: %s\n", what_s, err.c_str());
    ++fails;
}

// a tensor as GGUF lays one out
struct T {
    std::string name;
    std::vector<uint64_t> dims;   // 1-D for a direction; 2-D for the one bad file
    std::vector<float> data;
};

// a GGUF v3 file: string and int64 metadata keys, then these F32 tensors, their data at a 32-aligned start. The
// tensor offsets are filled after the header's size is known, walking it as a reader would.
std::vector<uint8_t> build(const std::vector<std::pair<std::string, std::string>>& strs,
                           const std::vector<std::pair<std::string, int64_t>>& ints, const std::vector<T>& ts) {
    std::vector<uint8_t> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    auto i64 = [&](int64_t v) { put(&v, 8); };
    auto str = [&](const std::string& s) { u64(s.size()); put(s.data(), s.size()); };
    u32(0x46554747u);
    u32(3);
    u64(ts.size());
    u64(strs.size() + ints.size());
    for (const auto& [k, v] : strs) { str(k); u32(8); str(v); }
    for (const auto& [k, v] : ints) { str(k); u32(11); i64(v); }   // GGUF int64 is type 11
    for (const auto& t : ts) {
        str(t.name);
        u32((uint32_t) t.dims.size());
        for (uint64_t d : t.dims) u64(d);
        u32(0);   // F32
        u64(0);   // offset: filled below
    }
    while (b.size() % 32) b.push_back(0);
    const size_t data_at = b.size();
    size_t p = 24;   // past the magic, the version, and the two counts
    for (size_t k = 0; k < strs.size() + ints.size(); ++k) {
        uint64_t len;
        std::memcpy(&len, &b[p], 8);
        p += 8 + (size_t) len;
        uint32_t ty;
        std::memcpy(&ty, &b[p], 4);   // the cursor is on the type here, as the reader's is
        p += 4;
        if (ty == 8) {
            uint64_t sl;
            std::memcpy(&sl, &b[p], 8);
            p += 8 + (size_t) sl;
        } else
            p += 8;   // an int64
    }
    // the offsets GGUF stores are from the data section's start, not absolute
    size_t off = 0;
    for (const auto& t : ts) {
        uint64_t len;
        std::memcpy(&len, &b[p], 8);
        p += 8 + (size_t) len;
        uint32_t nd;
        std::memcpy(&nd, &b[p], 4);
        p += 4 + 8 * nd;   // the dims
        p += 4;            // the type
        std::memcpy(&b[p], &off, 8);
        p += 8;
        off += t.data.size() * 4;
    }
    for (const auto& t : ts) for (float v : t.data) put(&v, 4);
    return b;
}

bool write_file(const std::string& path, const std::vector<uint8_t>& b) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char*) b.data(), (std::streamsize) b.size());
    return (bool) f;
}

// the metadata every good controlvector file here carries
const std::vector<std::pair<std::string, std::string>> GOOD_STR = {
    {"general.architecture", "controlvector"},
    {"controlvector.model_hint", "qwen4exp"},
};

// directions 1..count, `wide` floats each, unit: cos toward e0, sin toward e1, the rest zero
std::vector<T> dirs(int count, int wide) {
    std::vector<T> ts;
    for (int t = 1; t <= count; ++t) {
        T d;
        d.name = "direction." + std::to_string(t);
        d.dims = {(uint64_t) wide};
        d.data.assign((size_t) wide, 0.f);
        d.data[0] = std::cos(0.1f * (float) t);
        d.data[1] = std::sin(0.1f * (float) t);
        ts.push_back(std::move(d));
    }
    return ts;
}

CvecSpec spec(const std::string& path, double scale) { CvecSpec s; s.path = path; s.scale = scale; return s; }

Cvec must_load(const std::string& path, double scale, int wide, int nl, const char* what_s) {
    try {
        return load_cvec(spec(path, scale), "qwen4exp", wide, nl);
    } catch (const std::exception& e) {
        std::printf("NOTE: %s threw: %s\n", what_s, e.what());
        named(e.what(), path, what_s);
        return Cvec{};
    }
}
void refuses(const std::vector<uint8_t>& file, const std::string& path, const std::string& needle, const char* what_s,
             int wide, int nl) {
    write_file(path, file);
    try {
        load_cvec(spec(path, 1.0), "qwen4exp", wide, nl);
        ok(false, what_s);
    } catch (const std::exception& e) {
        std::printf("NOTE: %s threw: %s\n", what_s, e.what());
        named(e.what(), needle, what_s);
    }
}

void settings() {
    ::unsetenv("FNX_CVEC");
    ::unsetenv("FNX_CVEC_RANGE");
    CvecSpec s;
    ok(!cvec_from_env(s), "FNX_CVEC unset gives no vector");
    ::setenv("FNX_CVEC", "/tmp/d.gguf:1.78", 1);
    ok(cvec_from_env(s) && s.scale == 1.78 && s.path == "/tmp/d.gguf", "a good FNX_CVEC parses its file and scale");
    ::setenv("FNX_CVEC", "C:/models/d.gguf:1", 1);   // a path with a colon in it: the last one splits
    CvecSpec c2;
    ok(cvec_from_env(c2) && c2.path == "C:/models/d.gguf" && c2.scale == 1.0, "a colon inside the path");
    ::setenv("FNX_CVEC", "d.gguf", 1);
    try {
        cvec_from_env(s);
        ok(false, "a missing scale");
    } catch (const std::exception& e) { named(e.what(), "want FILE:SCALE", "a missing scale is refused"); }
    ::setenv("FNX_CVEC", "d.gguf:abc", 1);
    try {
        cvec_from_env(s);
        ok(false, "a scale that is not a number");
    } catch (const std::exception& e) { named(e.what(), "is not a number", "a bad scale is refused, naming the value"); }
    ::setenv("FNX_CVEC", "d.gguf:9", 1);
    try {
        cvec_from_env(s);
        ok(false, "a scale past 4");
    } catch (const std::exception& e) { named(e.what(), "from 0 to 4", "a scale out of range is refused"); }
    ::setenv("FNX_CVEC", "d.gguf:1", 1);
    ::setenv("FNX_CVEC_RANGE", "4:44", 1);
    CvecSpec r;
    ok(cvec_from_env(r) && r.range_lo == 4 && r.range_hi == 44, "a layer range parses");
    ::setenv("FNX_CVEC_RANGE", "44:4", 1);
    try {
        cvec_from_env(r);
        ok(false, "an inverted range");
    } catch (const std::exception& e) { named(e.what(), "START:END", "an inverted range is refused"); }
    ::unsetenv("FNX_CVEC");
    ::unsetenv("FNX_CVEC_RANGE");
}

void loading(const std::string& dir) {
    const int W = 16, NL = 8;   // a small stand-in model: wide residuals, few layers

    // the good file: directions 1..NL-1, each unit
    {
        const std::string p = dir + "/d.gguf";
        ok(write_file(p, build(GOOD_STR, {}, dirs(NL - 1, W))), "wrote d.gguf");
        Cvec c = must_load(p, 1.78, W, NL, "a good file loads");
        if (!c.on()) { std::printf("FAIL: the good file did not load; later checks skipped\n"); return; }
        ok(c.on() && c.lo == 1 && c.hi == NL - 1, "it names the blocks its directions cover");
        ok(!c.edits(0) && !c.edits(NL), "block 0 and past the last carry nothing");
        ok(c.edits(1) && c.edits(NL - 1), "the blocks it names are edited");
        ok(std::fabs(c.rows[(size_t) 1 * W] - std::cos(0.1f)) < 1e-6, "the rows are the file's bytes");
        // a range narrows what the file carries; one covering none of it is refused
        CvecSpec s = spec(p, 1.0);
        s.range_lo = 4, s.range_hi = 6;
        Cvec cr = load_cvec(s, "qwen4exp", W, NL);
        ok(cr.lo == 4 && cr.hi == 6 && !cr.edits(3) && cr.edits(4) && !cr.edits(7), "a range narrows, not extends");
        s.range_lo = 9, s.range_hi = 12;   // past what the file carries
        try {
            load_cvec(s, "qwen4exp", W, NL);
            ok(false, "a range outside the file");
        } catch (const std::exception& e) { named(e.what(), "covers none", "a range covering none of it is refused"); }
    }
    // no directions at all
    refuses(build(GOOD_STR, {}, {}), dir + "/e0.gguf", "no direction.N", "a file with no directions", W, NL);
    // direction.0 (llama.cpp reserves layer 0, and the reading has nothing to apply it at)
    {
        std::vector<T> ts = dirs(1, W);
        ts[0].name = "direction.0";
        refuses(build(GOOD_STR, {}, ts), dir + "/e1.gguf", "direction.0", "block 0's direction is refused", W, NL);
    }
    // past the model's last block
    {
        std::vector<T> ts = dirs(1, W);
        ts[0].name = "direction." + std::to_string(NL);
        refuses(build(GOOD_STR, {}, ts), dir + "/e1b.gguf", "direction." + std::to_string(NL),
                "a direction past the last block is refused", W, NL);
    }
    // not unit: the removal is exact only for a unit direction
    {
        std::vector<T> ts = dirs(3, W);
        ts[0].data[0] = 2.0f;
        ts[0].data[1] = 0.0f;
        refuses(build(GOOD_STR, {}, ts), dir + "/e2.gguf", "not unit", "a non-unit direction is refused", W, NL);
    }
    // the wrong width
    refuses(build(GOOD_STR, {}, dirs(3, 8)), dir + "/e3.gguf", "wide", "a direction of another width is refused", W, NL);
    // another model's hint
    {
        auto strs = GOOD_STR;
        strs[1] = {"controlvector.model_hint", "deepseek4"};
        refuses(build(strs, {}, dirs(3, W)), dir + "/e4.gguf", "model hint", "another model's hint is refused", W, NL);
    }
    // a model file, not a controlvector file
    {
        auto strs = GOOD_STR;
        strs[0] = {"general.architecture", "qwen4exp"};
        refuses(build(strs, {}, dirs(3, W)), dir + "/e5.gguf", "not controlvector", "a model GGUF is refused", W, NL);
    }
    // a 2-D tensor named direction.N: the same count of floats is not a direction
    {
        std::vector<T> ts = dirs(3, W);
        ts[0].dims = {(uint64_t) W, 1};
        refuses(build(GOOD_STR, {}, ts), dir + "/e6.gguf", "direction.1", "a matrix named direction.N is refused", W, NL);
    }
    // a layer_count that lies about the file
    {
        std::vector<std::pair<std::string, int64_t>> ints = {{"controlvector.layer_count", 4}};
        refuses(build(GOOD_STR, ints, dirs(3, W)), dir + "/e7.gguf", "layers and holds", "a lying layer_count is refused", W, NL);
    }
}

void the_real_file(int argc, char** argv) {
    if (argc < 2) {
        std::printf("note: no DIRECTION.gguf given: the real file's checks were not run\n");
        return;
    }
    const std::string path = argv[1];
    const int W = 2560, NL = 48;   // the model the feature is built for, as src/model.hpp names it
    Cvec c = must_load(path, 1.78, W, NL, "the real file loads");
    if (!c.on()) return;
    int kept = 0;
    for (char x : c.carried) kept += x != 0;
    ok(c.lo == 1 && c.hi == NL - 1 && kept == NL - 1, "and covers blocks 1..47, every one");
    std::printf("note: %s loaded: %d directions of %d, blocks %d..%d\n", path.c_str(), kept, W, c.lo, c.hi);
}

}  // namespace

int main(int argc, char** argv) {
    char tmpl[] = "/tmp/cvec_test_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    settings();
    loading(dir);
    the_real_file(argc, argv);
    if (fails) {
        std::printf("%d FAILED\n", fails);
        return 1;
    }
    std::printf("cvec_test: ok\n");
    return 0;
}

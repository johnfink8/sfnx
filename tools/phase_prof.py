"""Scratch profiling build of one kernel: copies fnx's src/ and tools/ into OUT, inserts clock64 stamps (thread 0 of
every block) at marker lines, builds kbench with a reader, runs it and prints the mean cycles between stamps per
block, at each T.

    phase_prof.py OUT KERNEL_NAME KBENCH_LINE MARKER_REGEX...
    phase_prof.py OUT KERNEL_NAME @engine:T MARKER_REGEX...

KBENCH_LINE: a substring of the printf in tools/kbench.cu after which the stamps are printed (e.g. "T=%d hc_read").
@engine:T instead builds fnx_run and runs 40 windows of T rows on the model (through run.sh): the stamps are the
kernel's last launch (the last layer that runs it, last window), printed per card.

Each MARKER_REGEX matches one source line inside the kernel; a stamp goes right after the first line that matches
it (searched from the kernel's start, in order). The kernel entry is stamp 0 and the end of the kernel source is
not stamped: give a marker on the last statement you care about."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

out, kernel, kline, markers = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
fnx = Path(__file__).resolve().parent.parent
if out.exists():
    shutil.rmtree(out)
shutil.copytree(fnx / "src", out / "src")
shutil.copytree(fnx / "tools", out / "tools")
for f in ("Makefile", "run.sh"):
    shutil.copy(fnx / f, out / f)
shutil.copytree(fnx / "ref", out / "ref", ignore=shutil.ignore_patterns("*.py"))
engine_T = int(kline.split(":")[1]) if kline.startswith("@engine:") else 0

cu_path = out / "src" / "kernels.cu"
# the kernel's file: kernels.cu or one of the headers it includes
kfile = next(f for f in sorted((out / "src").glob("*.cu*")) if re.search(rf"__global__[^\n]*\n?[^\n]*\b{kernel}\(", f.read_text()))
lines = kfile.read_text().split("\n")
start = next(i for i, l in enumerate(lines) if re.search(rf"\b{kernel}\(", l) and "__global__" in "".join(lines[max(0, i - 1):i + 1]))
brace = next(i for i in range(start, len(lines)) if lines[i].rstrip().endswith("{"))
import os
glob = os.environ.get("PP_GLOBAL") == "1"
clock = "fnx_gtime()" if glob else "clock64()"
stamp = "    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x < 1024) g_prof[blockIdx.x][{k}] = " + clock + ";"
inserts = [(brace, stamp.format(k=0))]
pos = brace
for k, m in enumerate(markers, start=1):
    pos = next(i for i in range(pos + 1, len(lines)) if re.search(m, lines[i]))
    inserts.append((pos, stamp.format(k=k)))
for i, text in sorted(inserts, reverse=True):
    lines.insert(i + 1, text)
kfile.write_text("\n".join(lines))
src = cu_path.read_text()
if "namespace fnx {\n\nnamespace {\n" not in src:
    raise SystemExit("kernels.cu: no place for g_prof")
src = src.replace("namespace fnx {\n\nnamespace {\n",
                  "namespace fnx {\n\nnamespace {\n__device__ unsigned long long g_prof[1024][16];\n"
                  "__device__ __forceinline__ unsigned long long fnx_gtime() {\n"
                  "    unsigned long long t;\n    asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t));\n    return t;\n}\n", 1)
src += """
namespace fnx {
void prof_read(unsigned long long* h) { cudaMemcpyFromSymbol(h, g_prof, sizeof(g_prof)); }
void prof_clear() { static unsigned long long z[1024][16]; cudaMemcpyToSymbol(g_prof, z, sizeof(z)); }
}
"""
cu_path.write_text(src)

PRINT = """
namespace fnx { void prof_read(unsigned long long* h); void prof_clear(); }
static void prof_print(const char* what, int T, int nstamps) {
    static unsigned long long h[1024][16];
    fnx::prof_read(&h[0][0]);
    double sum[16] = {};
    int n = 0;
    for (int b = 0; b < 1024; ++b) {
        bool all = true;   // blocks that reached every stamp (a stamp after an early return marks only the rest)
        for (int k = 0; k < nstamps; ++k) all = all && h[b][k];
        if (!all) continue;
        ++n;
        for (int k = 1; k < nstamps; ++k) sum[k] += (double) (h[b][k] - h[b][k - 1]);
    }
    std::printf("  %s T=%d, %d blocks, mean cycles per phase:", what, T, n);
    for (int k = 1; k < nstamps; ++k) std::printf(" %.0f", n ? sum[k] / n : 0.0);
    std::printf("\\n");
#ifdef PP_GLOBAL
    // one clock for every block: each stamp's earliest / median / latest over the blocks, ns from the first start
    unsigned long long t0 = ~0ull;
    for (int b = 0; b < 1024; ++b)
        if (h[b][0] && h[b][0] < t0) t0 = h[b][0];
    for (int k = 0; k < nstamps; ++k) {
        std::vector<unsigned long long> v;
        for (int b = 0; b < 1024; ++b)
            if (h[b][k]) v.push_back(h[b][k] - t0);
        std::sort(v.begin(), v.end());
        if (!v.empty()) std::printf("    stamp %d: %llu / %llu / %llu ns (%zu blocks)\\n", k, v.front(), v[v.size() / 2], v.back(), v.size());
    }
#endif
    fnx::prof_clear();
}"""
gg = Path(__file__).resolve().parents[1] / "llama.cpp/ggml"
flags = ["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_70", "-ccbin", "g++-14", "-Wno-deprecated-gpu-targets", f"-I{gg}/src/ggml-cuda", f"-I{gg}/src",
         f"-I{gg}/include", "--expt-relaxed-constexpr", f"-DNSTAMPS={len(markers) + 1}"] + (["-DPP_GLOBAL"] if glob else [])
if engine_T:
    fr = (out / "tools" / "fnx_run.cpp").read_text()
    fr = fr.replace('#include "../src/engine.hpp"', '#include "../src/engine.hpp"\n#include <cuda_runtime.h>\n' + PRINT, 1)
    hook = "        std::sort(wall.begin(), wall.end());"
    if fr.count(hook) != 1:
        raise SystemExit("fnx_run: no hook in rows mode")
    fr = fr.replace(hook, f"""        for (int d = 0; d < 3; ++d) {{
            cudaSetDevice(d);
            std::printf("card %d:", d);
            prof_print("{kernel}", T, NSTAMPS);
        }}
""" + hook)
    (out / "tools" / "fnx_run.cpp").write_text(fr)
    src_files = [str(out / "src" / f) for f in ("gguf.cpp", "model.cpp", "kernels.cu", "engine.cu")]
    r = subprocess.run(["/usr/local/cuda/bin/nvcc", *flags, "-o", str(out / "fnx_run"), str(out / "tools" / "fnx_run.cpp"), *src_files],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    model = Path(__file__).resolve().parents[1] / "models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
    r = subprocess.run([str(out / "run.sh"), str(out / "fnx_run"), str(model), "rows", str(out / "ref" / "ref1.bin"), str(engine_T), "40"],
                       capture_output=True, text=True, cwd=out)
    print("\n".join(l for l in r.stdout.split("\n") if "mean cycles" in l or "windows of" in l or "stamp " in l or l.startswith("card ")))
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])
    sys.exit(0)

kb = (out / "tools" / "kbench.cu").read_text()
kb = kb.replace("using namespace fnx;", "using namespace fnx;" + PRINT, 1)
n_hooks = len(re.findall(r"std::printf\(\"" + re.escape(kline), kb))
if n_hooks != 1:
    raise SystemExit(f"kbench: {n_hooks} printf lines start with {kline!r}")
kb = re.sub(r"(std::printf\(\"" + re.escape(kline) + r"[\s\S]*?\);\n)", lambda m: m.group(1) + f"            prof_print(\"{kernel}\", T, NSTAMPS);\n", kb)
(out / "tools" / "kbench.cu").write_text(kb)

cmd = ["/usr/local/cuda/bin/nvcc", *flags, "-o", str(out / "kb"), str(out / "tools" / "kbench.cu"), str(cu_path)]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode:
    raise SystemExit(r.stderr[-3000:])
r = subprocess.run([str(out / "kb")], capture_output=True, text=True, check=True)
for line in r.stdout.split("\n"):
    if line.startswith("T=") or "mean cycles" in line:
        print(line)

# fnx build: CUDA 12.9, sm_70, gcc-14 (prod's V100s; ARCH=86 for RTX 3090s, other arches below)
#   make            the engine and the picture encoder (bin/sfnx, bin/sfnx-vision; llama.cpp's libraries first if missing)
#   make install    the systemd unit (serve/sfnx.service.in for this checkout and user, port 8080), enabled at boot;
#                   CONFIG=path runs that config instead of serve/sfnx.json (it may live outside the checkout)
#   make test-bin   the test programs (bin/fnx_run, bin/batch_test, bin/cache_test, bin/sample_test, bin/gguf_test,
#                   bin/topk_test, bin/sfnx_input_test, bin/batch_plan_test, bin/stage_test,
#                   bin/pf_ring_plan_test, bin/rope_test, bin/card_check_test)
#   make ref        llama.cpp's reference tools (ref/fnx_ref, ref/fnx_logref, ref/img_logref, ref/mmid_bench)
#   make trace      the CUPTI kernel tracer for `tools/lab.sh trace` and the *_trace.sh scripts
#   LLAMA=path      llama.cpp at a37cc46 somewhere other than ./llama.cpp
#   ARCH=86         the cards' compute capability: 70 (V100, prod, the default), 86 (RTX 3090), or 80, 89, 90, 100, 120
#                   (build, never run)
#   CUDA=path       the CUDA toolkit, /usr/local/cuda by default: nvcc, CUPTI and llama.cpp's build all come from it
.DEFAULT_GOAL := build
.PHONY: build install test-bin ref trace
CUDA ?= /usr/local/cuda
NVCC := $(CUDA)/bin/nvcc
# 70 is prod's. 86 has run on RTX 3090s at a tester's (CUDA 12.9 there: with 12.8 and driver 595 the engine hung in
# cudaGraphInstantiate); its numbers are theirs (docs/verification.md). 80, 89, 90, 100 and 120 compile (every object,
# nvcc 12.9) and have never run; what a card must be beyond its arch the engine checks when it starts
# (docs/config.md, Other GPUs). 75 is refused here: the prompt path's recurrent kernel takes ~73 KB of shared memory a
# block (GCH_REC_SMEM, src/gdn_prompt.cuh) and Turing gives 64, and no Turing card has HC_READ_BLOCKS SMs anyway.
# build/arch holds the arch the objects in build/ were made for, as build/ncard holds the count, and build/cuda the nvcc
# and its version
ARCH ?= 70
# stripped, as " 86" from the environment would split the flags and never match its stamp; then one word, as $(filter)
# matches word by word and "86 90" would pass it
override ARCH := $(strip $(ARCH))
$(if $(word 2,$(ARCH)),$(error ARCH: one value, not '$(ARCH)'))
$(if $(filter 75,$(ARCH)),$(error ARCH: 75 (Turing) gives a block 64 KB of shared memory, and the prompt path's recurrent kernel takes ~73 KB))
$(if $(filter 70 80 86 89 90 100 120,$(ARCH)),,$(error ARCH: 70 (V100), 86 (RTX 3090), or 80, 89, 90, 100, 120 (build, never run); not '$(ARCH)'))
ref: ref/fnx_ref ref/fnx_logref ref/img_logref ref/mmid_bench
trace: tools/trace/libcupti_trace.so
CUPTI := $(CUDA)/extras/CUPTI
tools/trace/libcupti_trace.so: tools/trace/cupti_trace.cpp
	g++-14 -O2 -std=c++17 -shared -fPIC -I$(CUPTI)/include -I$(CUDA)/include -o $@ $< \
	    -L$(CUPTI)/lib64 -Wl,-rpath,$(CUPTI)/lib64 -lcupti -lpthread
# no built-in rules: with them make remakes an included build/kernels_p0.d by `%: %.o` from build/kernels_p0.d.o, which
# the kernels_p% rule then compiles with FNX_KPART=0.d ("floating constant in preprocessor expression")
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
build: bin/sfnx bin/sfnx-vision
test-bin: bin/fnx_run bin/batch_test bin/cache_test bin/sample_test bin/gguf_test bin/topk_test bin/sfnx_input_test bin/batch_plan_test \
          bin/stage_test bin/pf_ring_plan_test bin/rope_test bin/card_check_test
CONFIG ?= serve/sfnx.json
# The config is checked as the front end checks it before loading anything (front.check_config), so a config it would
# refuse stops the install rather than a boot; what needs the model (the context against its trained length) the engine
# refuses when it starts. Its path is made absolute but a link is kept a link (repointing it then
# switches configs at the next restart), and it may hold only what sed and systemd's ExecStart take as is. The unit is
# written whole to build/ first and moved into place only once nothing in it is left unfilled.
install: build
	cfg=$$(realpath -s "$(CONFIG)") && [ -f "$$cfg" ] || { echo "install: no config $(CONFIG)" >&2; exit 1; }; \
	for p in "$$cfg" "$(CURDIR)"; do case "$$p" in *[!A-Za-z0-9._/+-]*) \
	    echo "install: $$p has characters a unit's ExecStart cannot hold as is (letters, digits, . _ / + - only)" >&2; exit 1;; esac; done; \
	.venv/bin/python -m front.check_config "$$cfg" && \
	sed -e "s|@ROOT@|$(CURDIR)|g" -e "s|@CONFIG@|$$cfg|" -e "s|@USER@|$$(id -un)|" -e "s|@GROUP@|$$(id -gn)|" \
	    serve/sfnx.service.in > build/sfnx.service && \
	if grep -n '@[A-Z]*@' build/sfnx.service; then echo "install: build/sfnx.service has unfilled fields" >&2; exit 1; fi && \
	sudo install -m 644 build/sfnx.service /etc/systemd/system/sfnx.service
	sudo systemctl daemon-reload
	sudo systemctl enable sfnx
	@echo "install: the unit runs $(CONFIG); a running sfnx keeps its old config until sudo systemctl restart sfnx"

# llama.cpp at a37cc46 (its ggml headers; its libraries for the picture encoder): a checkout in llama.cpp/, or LLAMA=path
LLAMA ?= $(CURDIR)/llama.cpp
GG := $(LLAMA)/ggml
NVFLAGS := -DFNX_DIR='"$(CURDIR)"' -O3 --use_fast_math -std=c++17 -arch=sm_$(ARCH) -ccbin g++-14 -Wno-deprecated-gpu-targets -I$(GG)/src/ggml-cuda -I$(GG)/src -I$(GG)/include --expt-relaxed-constexpr
# the cards every layer is split across (src/model.hpp): 3 (prod), or 2, 4, 6 or 8 (make FNX_NCARD=4 ...; the counts
# the model's shapes divide by, src/card_split.hpp; above three untested on hardware). build/ncard holds the count the
# objects in build/ were made for; a different count rewrites it, which rebuilds them all. It is written while make
# reads this file, so any make in this checkout restamps it to its count, -n and `make ref` too
FNX_NCARD ?= 3
# stripped and one word, as ARCH is
override FNX_NCARD := $(strip $(FNX_NCARD))
$(if $(word 2,$(FNX_NCARD)),$(error FNX_NCARD: one value, not '$(FNX_NCARD)'))
$(if $(filter 2 3 4 6 8,$(FNX_NCARD)),,$(error FNX_NCARD: 2, 3, 4, 6 or 8 (the 24 query heads and 48 value heads split evenly, a 32-value FF block on every card, at most 8 for CUDA's peers a card), not '$(FNX_NCARD)'))
NVFLAGS += -DFNX_NCARD=$(FNX_NCARD) -DFNX_ARCH=$(ARCH)
# the model (src/model.hpp): fnx (Flash-Next, prod) or 27b (Qwen3.8-27B: src/s27.*, bin/s27_run). build/model holds
# it as build/ncard holds the count; the 27B has run on three cards only, so other counts are refused for it
FNX_MODEL ?= fnx
override FNX_MODEL := $(strip $(FNX_MODEL))
$(if $(filter fnx 27b,$(FNX_MODEL)),,$(error FNX_MODEL: fnx or 27b, not '$(FNX_MODEL)'))
$(if $(and $(filter 27b,$(FNX_MODEL)),$(filter-out 3,$(FNX_NCARD))),$(error FNX_MODEL=27b: three cards only (its K/V heads are placed for three), not FNX_NCARD=$(FNX_NCARD)))
FNX_27B := $(if $(filter 27b,$(FNX_MODEL)),1,0)
NVFLAGS += -DFNX_27B=$(FNX_27B)
# the stamps, written with make's own $(file) (4.2 and later), not a shell, so no setting's text runs as one; and only
# when their text changes, as a rewritten stamp rebuilds everything. stamp(file,text)
stamp = $(if $(and $(findstring x$(2)x,x$(file <$(1))x),$(findstring x$(file <$(1))x,x$(2)x)),,$(file >$(1),$(2)))
$(shell mkdir -p build)
$(call stamp,build/ncard,$(FNX_NCARD))
$(call stamp,build/arch,$(ARCH))
$(call stamp,build/model,$(FNX_MODEL))
# nvcc's last version line, "none" when there is none: the CPU tests still build, anything CUDA fails at its first nvcc.
# (Running it takes CUDA's path through a shell, as every recipe here does)
NVCC_VERSION := $(shell "$(NVCC)" --version 2>/dev/null | tail -1)
NVCC_ID := $(NVCC) $(or $(NVCC_VERSION),none)
$(if $(NVCC_VERSION),,$(if $(wildcard $(NVCC)),$(warning nvcc at $(NVCC) gave no version: anything CUDA will fail),\
    $(warning no nvcc at $(NVCC) (CUDA=$(CUDA)): only the CPU tests can build)))
$(call stamp,build/cuda,$(NVCC_ID))
STAMPS := build/ncard build/arch build/cuda build/model

bin/gguf_test: tools/gguf_test.cpp src/gguf.cpp src/gguf.hpp src/rope_gguf.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -o $@ tools/gguf_test.cpp src/gguf.cpp
bin/stage_test: tools/stage_test.cpp src/gguf.cpp src/gguf.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -o $@ tools/stage_test.cpp src/gguf.cpp -lpthread
bin/sfnx_input_test: tools/sfnx_input_test.cpp tools/sfnx_input.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -pthread -o $@ tools/sfnx_input_test.cpp
# the batched drafts' row plan, on the CPU alone
bin/batch_plan_test: tools/batch_plan_test.cpp src/batch_plan.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -o $@ tools/batch_plan_test.cpp
# the prompt path's ring plan against the two- and three-card sequences, and every supported count's sum, on the CPU
bin/pf_ring_plan_test: tools/pf_ring_plan_test.cpp src/pf_ring_plan.hpp src/card_split.hpp src/model.hpp src/gguf.hpp build/ncard build/model
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -DFNX_NCARD=$(FNX_NCARD) -DFNX_27B=$(FNX_27B) -o $@ tools/pf_ring_plan_test.cpp
# YaRN's band, angles and magnitude against ggml's, on the CPU alone
bin/rope_test: tools/rope_test.cpp src/rope.hpp
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -o $@ tools/rope_test.cpp
# the start-up card checks against real cards' numbers and sysfs text, on the CPU alone
bin/card_check_test: tools/card_check_test.cpp src/card_check.hpp src/peer_probe.hpp src/model.hpp src/card_split.hpp src/gguf.hpp build/ncard build/model
	mkdir -p bin && g++-14 -O2 -std=c++17 -Wall -DFNX_NCARD=$(FNX_NCARD) -DFNX_27B=$(FNX_27B) -o $@ tools/card_check_test.cpp

# one object per source in build/ (headers tracked by nvcc's own dependency files), so a change recompiles only the
# objects it touches; every engine program and the lab's library link the same objects (position independent).
# kernels.cu compiles as KPARTS objects in parallel (make -j), each with its share of the launchers (FNX_KPART)
KPARTS := 0 1 2 3 4 5 6 7 8
KERN_OBJ := $(foreach p,$(KPARTS),build/kernels_p$(p).o)
ENG_OBJ := build/gguf.o build/model.o build/store.o $(KERN_OBJ) build/moe_mma.o build/engine.o build/peer_probe.o build/prefill.o build/session.o build/batch.o
# the 27B's Engine (engine.hpp's interface) is its own: s27_engine.cu and its prompt path
ifeq ($(FNX_27B),1)
ENG_OBJ := build/gguf.o build/model.o build/store.o $(KERN_OBJ) build/moe_mma.o build/s27_engine.o build/peer_probe.o build/s27_prefill.o build/session.o build/batch.o
endif
PIC := -Xcompiler -fPIC
build/kernels_p%.o: src/kernels.cu $(STAMPS)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -DFNX_KPART=$* -MD -MF build/kernels_p$*.d -c -o $@ $<
build/%.o: src/%.cu $(STAMPS)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/$*.d -c -o $@ $<
build/%.o: src/%.cpp $(STAMPS)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/$*.d -c -o $@ $<
build/tool_%.o: tools/%.cpp $(STAMPS)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/tool_$*.d -c -o $@ $<
build/tool_%.o: tools/%.cu $(STAMPS)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(PIC) -MD -MF build/tool_$*.d -c -o $@ $<
-include $(wildcard build/*.d)

bin/load_probe: build/tool_load_probe.o build/gguf.o build/model.o build/store.o
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^ -lcuda
# the 27B's own programs (make FNX_MODEL=27b), and the lab, which holds Flash-Next only (the 27B loads plain)
S27_BIN := bin/s27_run bin/s27_probe bin/s27_ring_test
FNX_BIN := lab bin/fnx_lab bin/fnx_hold build/libfnx_lab.so
$(if $(and $(filter 0,$(FNX_27B)),$(filter $(S27_BIN),$(MAKECMDGOALS))),$(error $(filter $(S27_BIN),$(MAKECMDGOALS)): make FNX_MODEL=27b))
$(if $(and $(filter 1,$(FNX_27B)),$(filter $(FNX_BIN),$(MAKECMDGOALS))),$(error $(filter $(FNX_BIN),$(MAKECMDGOALS)): Flash-Next's, not FNX_MODEL=27b's))
bin/s27_run: build/tool_s27_run.o $(ENG_OBJ)
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^ -lcublas -lcuda -lpthread
bin/kbench bin/s27_probe bin/s27_ring_test bin/moe_mma_bench bin/topk_test: bin/%: build/tool_%.o $(KERN_OBJ) build/moe_mma.o
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^
bin/sfnx: build/tool_sfnx.o $(ENG_OBJ)
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^ -lcublas -lcuda -lpthread
bin/fnx_run bin/sample_test bin/cache_test bin/lanes_test bin/batch_test: bin/%: build/tool_%.o $(ENG_OBJ)
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -o $@ $^ -lcublas -lcuda

# the lab (tools/fnx_lab.cpp): bin/fnx_lab keeps the model loaded and runs fnx_run's commands from build/libfnx_lab.so,
# which holds the engine and is reloaded after a rebuild; both use the shared CUDA runtime, and the lab exports the
# model's code (gguf, model) to the library
LAB_OBJ := $(KERN_OBJ) build/moe_mma.o build/engine.o build/peer_probe.o build/prefill.o build/session.o build/batch.o build/pic/tool_fnx_run.o
build/pic/tool_fnx_run.o: tools/fnx_run.cpp $(STAMPS)
	@mkdir -p build/pic
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -DFNX_LAB_LIB -MD -MF build/pic/tool_fnx_run.d -c -o $@ $<
-include $(wildcard build/pic/*.d)
build/libfnx_lab.so: $(LAB_OBJ)
	$(NVCC) $(NVFLAGS) -shared -cudart shared -o $@.new $^ -lcublas -lcuda && mv -f $@.new $@
bin/fnx_lab: build/tool_fnx_lab.o build/gguf.o build/model.o build/store.o build/libfnx_lab.so
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -cudart shared -Xlinker -export-dynamic -o $@ build/tool_fnx_lab.o build/gguf.o build/model.o build/store.o -ldl -lcuda
.PHONY: lab
# bin/fnx_hold: the lab's holder (the weights in shared memory, src/store.hpp; tools/lab.sh starts it)
bin/fnx_hold: build/tool_fnx_hold.o build/gguf.o build/model.o build/store.o
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) -cudart shared -o $@ $^ -lcuda
lab: bin/fnx_lab bin/fnx_hold build/libfnx_lab.so
# the loader sources' hash, built into the holder and the worker: the holder refuses a worker built from other ones
# (tools/lab_proto.hpp). A new hash names a new stamp file, which rebuilds both objects. A count other than 3 joins the
# hash (a two-card worker refuses a three-card holder); three cards' hash is the sources' alone, so prod's holder
# stamp follows only the loader's code
LOADER_SRC := src/model.cpp src/model.hpp src/card_split.hpp src/gguf.cpp src/gguf.hpp src/store.cpp src/store.hpp tools/lab_proto.hpp
$(if $(filter-out $(wildcard $(LOADER_SRC)),$(LOADER_SRC)),$(error loader sources missing: $(filter-out $(wildcard $(LOADER_SRC)),$(LOADER_SRC))))
LOADER_REV := $(shell { cat $(LOADER_SRC); [ $(FNX_NCARD) = 3 ] || echo ncard $(FNX_NCARD); } | sha1sum | cut -c1-16)
build/loader_rev.$(LOADER_REV):
	@mkdir -p build && touch $@
build/tool_fnx_hold.o build/tool_fnx_lab.o: build/loader_rev.$(LOADER_REV)
build/tool_fnx_hold.o build/tool_fnx_lab.o: NVFLAGS += -DFNX_LOADER_REV='"$(LOADER_REV)"'

# the image encoder (vision/): llama.cpp's mtmd, built there with CUDA for ARCH
LU := $(LLAMA)
# llama.cpp's libraries are built once (the rule below has no prerequisites), so ones built for another arch or with
# another nvcc stop here, not at the first picture ("no kernel image") or as kernels the driver compiles from PTX at
# load. The arch is the cubins' in libggml-cuda.so itself; nvcc is the one CMake found when it configured, by its path
# (links followed, so another install at the same version shows) and its version (so a toolkit swapped under the same
# path shows); a configure not followed by a build does not. Each input missing
# says so by name rather than as a mismatch. A second worktree's llama.cpp is the main checkout's: one arch and
# toolkit a box. The encoder is rebuilt with the engine's arch (build/arch)
.PHONY: llama-check
llama-check: $(LU)/build/bin/libmtmd.so
	@lib=$(LU)/build/bin/libggml-cuda.so; dump=$(CUDA)/bin/cuobjdump; \
	[ -x "$$dump" ] || { echo "llama-check: no cuobjdump at $$dump: CUDA=$(CUDA) is not a CUDA toolkit" >&2; exit 1; }; \
	[ -f "$$lib" ] || { echo "llama-check: no $$lib: llama.cpp in $(LU)/build was built without CUDA (GGML_CUDA=ON)" >&2; exit 1; }; \
	elf=$$("$$dump" --list-elf "$$lib") || { echo "llama-check: $$dump --list-elf $$lib failed" >&2; exit 1; }; \
	got=$$(printf '%s\n' "$$elf" | sed -n 's/.*\.\(sm_[0-9a-z]*\)\.cubin$$/\1/p' | sort -u | tr '\n' ' '); \
	[ "$$got" = "sm_$(ARCH) " ] || { echo "llama-check: $$lib holds kernels for '$$got', this build is for sm_$(ARCH): reconfigure llama.cpp with -DCMAKE_CUDA_ARCHITECTURES=$(ARCH) and rebuild it (cmake --build $(LU)/build --target mtmd llama)" >&2; exit 1; }; \
	set -- $(LU)/build/CMakeFiles/*/CMakeCUDACompiler.cmake; \
	[ -f "$$1" ] || { echo "llama-check: no $(LU)/build/CMakeFiles/*/CMakeCUDACompiler.cmake: CMake never configured CUDA there" >&2; exit 1; }; \
	[ $$# = 1 ] || { echo "llama-check: $$# CMake versions configured $(LU)/build ($$*): which nvcc built it is unknown; move it aside and make builds it again" >&2; exit 1; }; \
	was=$$(sed -n 's/^set(CMAKE_CUDA_COMPILER_VERSION "\(.*\)")$$/\1/p' "$$1"); \
	[ -n "$$was" ] || { echo "llama-check: no CMAKE_CUDA_COMPILER_VERSION in $$1" >&2; exit 1; }; \
	[ -x "$(NVCC)" ] || { echo "llama-check: no nvcc at $(NVCC)" >&2; exit 1; }; \
	cc=$$(sed -n 's/^set(CMAKE_CUDA_COMPILER "\(.*\)")$$/\1/p' "$$1"); \
	[ -n "$$cc" ] || { echo "llama-check: no CMAKE_CUDA_COMPILER in $$1" >&2; exit 1; }; \
	ccr=$$(realpath -m "$$cc") && [ -n "$$ccr" ] || { echo "llama-check: realpath -m $$cc failed" >&2; exit 1; }; \
	nvr=$$(realpath "$(NVCC)") && [ -n "$$nvr" ] || { echo "llama-check: realpath $(NVCC) failed" >&2; exit 1; }; \
	[ "$$ccr" = "$$nvr" ] || { echo "llama-check: llama.cpp in $(LU)/build was configured with $$cc ($$ccr), this build's nvcc is $(NVCC) ($$nvr). CMake keeps the compiler it first found: move $(LU)/build aside and make builds it again with $(NVCC)" >&2; exit 1; }; \
	ver=$$("$(NVCC)" --version) || { echo "llama-check: $(NVCC) --version failed" >&2; exit 1; }; \
	now=$$(printf '%s\n' "$$ver" | sed -n 's/.*, V\([0-9.]*\)$$/\1/p'); \
	[ -n "$$now" ] || { echo "llama-check: no version in $(NVCC) --version: '$$ver'" >&2; exit 1; }; \
	[ "$$was" = "$$now" ] || { echo "llama-check: llama.cpp in $(LU)/build was configured with nvcc $$was ($$1), this build's $(NVCC) is $$now. CMake keeps the compiler it first found: move $(LU)/build aside and make builds it again with $(NVCC)" >&2; exit 1; }
bin/sfnx-vision: vision/sfnx_vision.cpp vision/*.h $(LU)/build/bin/libmtmd.so build/arch | llama-check
	mkdir -p bin && g++-14 -O2 -std=c++17 -I$(LU)/include -I$(GG)/include -I$(LU)/tools/mtmd -o $@ vision/sfnx_vision.cpp \
	    -L$(LU)/build/bin -lmtmd -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,$(LU)/build/bin -lpthread

# the llama.cpp reference tools (ref/): logits and tensor dumps the exactness checks compare against
LLFLAGS = -O2 -std=c++17 -I$(LU)/include -I$(GG)/include -L$(LU)/build/bin -Wl,-rpath,$(LU)/build/bin
ref/fnx_ref ref/fnx_logref: ref/%: ref/%.cpp ref/logref_yarn.hpp src/env.hpp $(LU)/build/bin/libmtmd.so | llama-check
	g++-14 $(LLFLAGS) -o $@ $< -lllama -lggml -lggml-base
ref/img_logref: ref/img_logref.cpp ref/logref_yarn.hpp src/env.hpp $(LU)/build/bin/libmtmd.so | llama-check
	g++-14 $(LLFLAGS) -I$(LU)/tools/mtmd -o $@ $< -lmtmd -lllama -lggml -lggml-base
ref/mmid_bench: ref/mmid_bench.cpp $(LU)/build/bin/libmtmd.so | llama-check
	g++-14 $(LLFLAGS) -o $@ $< -lggml -lggml-base -lggml-cuda

# llama.cpp's libraries, built in its own tree with CUDA for ARCH
$(LU)/build/bin/libmtmd.so:
	test -d $(LU)/.git || { echo "clone llama.cpp at a37cc46 into $(LU) first (or make LLAMA=path)" >&2; exit 1; }
	cmake -S $(LU) -B $(LU)/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$(ARCH) -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
	    -DCMAKE_CUDA_COMPILER=$(NVCC)
	cmake --build $(LU)/build --target mtmd llama

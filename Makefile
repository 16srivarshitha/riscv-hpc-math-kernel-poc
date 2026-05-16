# ============================================================================
# Makefile — RISC-V HPC Kernel Optimisation Sandbox
#
# Targets
#   all          build shared library + benchmark binary (default)
#   scalar       build with only scalar code (x86 / any host)
#   run          build and run on host (scalar fallback)
#   qemu-run     cross-compile for RV64GCV and run under QEMU
#   clean        remove build artefacts
#
# Toolchain selection
#   Native RISC-V  : CXX = riscv64-linux-gnu-g++
#   Cross-compile  : override from command line
#   Host fallback  : uses system c++ (scalar only, for CI / development)
#
# Usage examples
#   make                           # host build, scalar only
#   make CXX=riscv64-linux-gnu-g++ # cross-compile with RVV
#   make qemu-run                  # cross + QEMU
# ============================================================================

#  Compiler 
CXX       ?= c++
AR        ?= ar

#  Directories 
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build
OUT_DIR   := lib

#  Source files 
SCALAR_SRCS := $(SRC_DIR)/gemm_scalar.cpp \
               $(SRC_DIR)/hal_dispatch.cpp

RVV_SRCS    := $(SRC_DIR)/gemm_rvv.cpp   # contains fft_rvv + poisson_rvv too

ALL_SRCS    := $(SCALAR_SRCS) $(RVV_SRCS)
MAIN_SRC    := $(SRC_DIR)/main.cpp

#  Flags 
COMMON_FLAGS := -std=c++17 -Wall -Wextra -O3 -ffast-math \
                -I$(INC_DIR)

# RVV-specific: target RV64GCV with double-precision FP ABI.
# Enables __riscv_v preprocessor guard used in gemm_rvv.cpp and dispatch.h.
RVV_FLAGS    := -march=rv64gcv -mabi=lp64d

# Scalar-only: compile without any RISC-V extension.
# Used for host testing (x86, aarch64, …) and CI.
SCALAR_FLAGS :=

# Shared library flags
SO_FLAGS     := -shared -fPIC

#  Output names 
LIBNAME     := libhpc_kernels
SO          := $(OUT_DIR)/$(LIBNAME).so
STATIC      := $(OUT_DIR)/$(LIBNAME).a
BENCH_BIN   := $(BUILD_DIR)/bench

#  Auto-detect RISC-V toolchain 
IS_RISCV    := $(shell $(CXX) -dumpmachine 2>/dev/null | grep -c riscv)

ifeq ($(IS_RISCV),1)
  ARCH_FLAGS := $(RVV_FLAGS)
  $(info [Makefile] RISC-V toolchain detected — enabling RVV)
else
  ARCH_FLAGS := $(SCALAR_FLAGS)
  $(info [Makefile] Non-RISC-V toolchain — scalar fallback only)
endif

CXXFLAGS := $(COMMON_FLAGS) $(ARCH_FLAGS)

#  Object files 
SCALAR_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SCALAR_SRCS))
RVV_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(RVV_SRCS))
ALL_OBJS    := $(SCALAR_OBJS) $(RVV_OBJS)
MAIN_OBJ    := $(BUILD_DIR)/main.o

#  Default target 
.PHONY: all scalar run qemu-run clean dirs

all: dirs $(SO) $(BENCH_BIN)
	@echo ""
	@echo "  Build complete."
	@echo "  Shared library : $(SO)"
	@echo "  Benchmark      : $(BENCH_BIN)"

scalar: dirs
	$(MAKE) ARCH_FLAGS="$(SCALAR_FLAGS)" all

#  Directory creation 
dirs:
	@mkdir -p $(BUILD_DIR) $(OUT_DIR)

#  Compile rules 
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC) | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

#  Link shared library 
$(SO): $(ALL_OBJS)
	$(CXX) $(SO_FLAGS) $(CXXFLAGS) $^ -o $@ -lm
	@echo "  [SO]  $@"

#  Link benchmark binary 
$(BENCH_BIN): $(MAIN_OBJ) $(SO)
	$(CXX) $(CXXFLAGS) $< -L$(OUT_DIR) -lhpc_kernels -Wl,-rpath,$(OUT_DIR) -lm -o $@
	@echo "  [BIN] $@"

#  Run on host (scalar fallback) 
run: all
	@echo ""
	@echo " Running benchmark (host / scalar fallback) "
	LD_LIBRARY_PATH=$(OUT_DIR) $(BENCH_BIN)

#  Cross-compile + QEMU 
# Requires: riscv64-linux-gnu-g++  and  qemu-riscv64 in PATH.
# QEMU note: RVV intrinsics run slower under emulation due to
# per-instruction overhead.  Results validate correctness, not performance.
RISCV_CXX  ?= riscv64-linux-gnu-g++
QEMU       ?= qemu-riscv64
QEMU_FLAGS ?= -cpu rv64,v=true,vlen=512,elen=64,vext_spec=v1.0

qemu-run:
	@echo " Cross-compiling for RV64GCV "
	$(MAKE) CXX=$(RISCV_CXX) AR=riscv64-linux-gnu-ar all
	@echo ""
	@echo " Running under QEMU (vlen=512) "
	$(QEMU) $(QEMU_FLAGS) \
	    -E LD_LIBRARY_PATH=$(OUT_DIR) \
	    $(BENCH_BIN)

#  Clean 
clean:
	rm -rf $(BUILD_DIR) $(OUT_DIR)
	@echo "  Cleaned."
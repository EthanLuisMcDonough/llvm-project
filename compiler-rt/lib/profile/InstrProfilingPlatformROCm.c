//===- InstrProfilingPlatformROCm.c - Profile data ROCm platform ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingPort.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ProcessDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex);

static int IsVerboseMode() {
  static int IsVerbose = -1;
  if (IsVerbose == -1)
    IsVerbose = getenv("LLVM_PROFILE_VERBOSE") != NULL;
  return IsVerbose;
}

/* -------------------------------------------------------------------------- */
/*  Dynamic loading of HIP runtime symbols                                   */
/* -------------------------------------------------------------------------- */

typedef int (*hipMemcpyFromSymbolTy)(void *, const void *, size_t, size_t, int);
typedef int (*hipGetSymbolAddressTy)(void **, const void *);
typedef int (*hipMemcpyTy)(void *, void *, size_t, int);
typedef int (*hipModuleGetGlobalTy)(void **, size_t *, void *, const char *);

static hipMemcpyFromSymbolTy pHipMemcpyFromSymbol = NULL;
static hipGetSymbolAddressTy pHipGetSymbolAddress = NULL;
static hipMemcpyTy pHipMemcpy = NULL;
static hipModuleGetGlobalTy pHipModuleGetGlobal = NULL;

/* -------------------------------------------------------------------------- */
/*  Device-to-host copies                                                     */
/*  Keep HIP-only to avoid an HSA dependency.                                 */
/* -------------------------------------------------------------------------- */

static void EnsureHipLoaded(void) {
  static int Initialized = 0;
  if (Initialized)
    return;
  Initialized = 1;

  void *Handle = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
  if (!Handle) {
    fprintf(stderr, "compiler-rt: failed to open libamdhip64.so: %s\n",
            dlerror());
    return;
  }

  pHipMemcpyFromSymbol =
      (hipMemcpyFromSymbolTy)dlsym(Handle, "hipMemcpyFromSymbol");
  pHipGetSymbolAddress =
      (hipGetSymbolAddressTy)dlsym(Handle, "hipGetSymbolAddress");
  pHipMemcpy = (hipMemcpyTy)dlsym(Handle, "hipMemcpy");
  pHipModuleGetGlobal =
      (hipModuleGetGlobalTy)dlsym(Handle, "hipModuleGetGlobal");
}

/* -------------------------------------------------------------------------- */
/*  Public wrappers that forward to the loaded HIP symbols                   */
/* -------------------------------------------------------------------------- */

static int hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes,
                               size_t offset, int kind) {
  EnsureHipLoaded();
  return pHipMemcpyFromSymbol
             ? pHipMemcpyFromSymbol(dst, symbol, sizeBytes, offset, kind)
             : -1;
}

static int hipGetSymbolAddress(void **devPtr, const void *symbol) {
  EnsureHipLoaded();
  return pHipGetSymbolAddress ? pHipGetSymbolAddress(devPtr, symbol) : -1;
}

static int hipMemcpy(void *dest, void *src, size_t len, int kind /*2=DToH*/) {
  EnsureHipLoaded();
  return pHipMemcpy ? pHipMemcpy(dest, src, len, kind) : -1;
}

/* Copy from device to host using HIP.
 * This requires that the device section symbols are registered with CLR,
 * otherwise hipMemcpy may attempt a CPU path and crash. */
static int memcpyDeviceToHost(void *Dst, void *Src, size_t Size) {
  return hipMemcpy(Dst, Src, Size, 2 /* DToH */);
}

static int hipModuleGetGlobal(void **DevPtr, size_t *Bytes, void *Module,
                              const char *Name) {
  EnsureHipLoaded();
  return pHipModuleGetGlobal ? pHipModuleGetGlobal(DevPtr, Bytes, Module, Name)
                             : -1;
}

/* -------------------------------------------------------------------------- */
/*  Dynamic module tracking                                                   */
/* -------------------------------------------------------------------------- */

#define MAX_DYNAMIC_MODULES 256

typedef struct {
  void *ModulePtr; /* hipModule_t returned by hipModuleLoad            */
  void *DeviceVar; /* address of __llvm_offload_prf in this module     */
  int Processed;   /* 0 = not yet collected, 1 = data already copied   */
} OffloadDynamicModuleInfo;

static OffloadDynamicModuleInfo DynamicModules[MAX_DYNAMIC_MODULES];
static int NumDynamicModules = 0;

/* -------------------------------------------------------------------------- */
/*  Registration / un-registration helpers                                   */
/* -------------------------------------------------------------------------- */

void __llvm_profile_offload_register_dynamic_module(int ModuleLoadRc,
                                                    void **Ptr) {
  if (IsVerboseMode())
    PROF_NOTE("Registering loaded module %d: rc=%d, module=%p\n",
              NumDynamicModules, ModuleLoadRc, *Ptr);

  if (ModuleLoadRc)
    return;

  if (NumDynamicModules >= MAX_DYNAMIC_MODULES) {
    PROF_ERR("Too many dynamic modules registered. Maximum is %d.\n",
             MAX_DYNAMIC_MODULES);
    return;
  }

  OffloadDynamicModuleInfo *Info = &DynamicModules[NumDynamicModules++];
  Info->ModulePtr = *Ptr;
  Info->DeviceVar = NULL;
  Info->Processed = 0;

  size_t Bytes = 0;
  if (hipModuleGetGlobal(&Info->DeviceVar, &Bytes, *Ptr,
                         "__llvm_offload_prf") != 0) {
    PROF_WARN("Failed to get symbol __llvm_offload_prf for module %p\n", *Ptr);
    /* Leave DeviceVar NULL so later code can recognise the failure */
    return;
  }

  if (IsVerboseMode())
    PROF_NOTE("Module %p: Device profile var %p\n", *Ptr, Info->DeviceVar);
}

void __llvm_profile_offload_unregister_dynamic_module(void *Ptr) {
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *Info = &DynamicModules[i];

    if (Info->ModulePtr == Ptr) {
      if (IsVerboseMode())
        PROF_NOTE("Unregistering module %p (DeviceVar=%p, Processed=%d)\n",
                  Info->ModulePtr, Info->DeviceVar, Info->Processed);

      if (Info->Processed) {
        PROF_WARN("Module %p has already been unregistered or processed\n",
                  Ptr);
        return;
      }

      if (Info->DeviceVar) {
        // Use module index as TU index for dynamic modules
        // to ensure each module gets a unique profile file
        if (ProcessDeviceOffloadPrf(Info->DeviceVar, i) == 0)
          Info->Processed = 1;
        else
          PROF_WARN(
              "Failed to process profile data for module %p on unregister\n",
              Ptr);
      } else {
        PROF_WARN("Module %p has no device profile variable to process\n", Ptr);
      }
      return;
    }
  }

  if (IsVerboseMode())
    PROF_WARN("Unregister called for unknown module %p\n", Ptr);
}

#define MAX_SHADOW_VARIABLES 256
static void *OffloadShadowVariables[MAX_SHADOW_VARIABLES];
static int NumShadowVariables = 0;

void __llvm_profile_offload_register_shadow_variable(void *ptr) {
  if (NumShadowVariables >= MAX_SHADOW_VARIABLES) {
    PROF_ERR("Too many shadow variables registered. Maximum is %d.\n",
             MAX_SHADOW_VARIABLES);
    return;
  }
  if (IsVerboseMode())
    PROF_NOTE("Registering shadow variable %d: %p\n", NumShadowVariables, ptr);
  OffloadShadowVariables[NumShadowVariables++] = ptr;
}

#define MAX_SECTION_SHADOW_VARIABLES 1024
static void *OffloadSectionShadowVariables[MAX_SECTION_SHADOW_VARIABLES];
static int NumSectionShadowVariables = 0;

void __llvm_profile_offload_register_section_shadow_variable(void *ptr) {
  if (NumSectionShadowVariables >= MAX_SECTION_SHADOW_VARIABLES) {
    PROF_ERR("Too many section shadow variables registered. Maximum is %d.\n",
             MAX_SECTION_SHADOW_VARIABLES);
    return;
  }
  if (IsVerboseMode())
    PROF_NOTE("Registering section shadow variable %d: %p\n",
              NumSectionShadowVariables, ptr);
  OffloadSectionShadowVariables[NumSectionShadowVariables++] = ptr;
}

static int ProcessDeviceOffloadPrf(void *DeviceOffloadPrf, int TUIndex) {
  void *HostOffloadPrf[8];

  if (IsVerboseMode())
    PROF_NOTE("HostOffloadPrf buffer size: %zu bytes\n",
              sizeof(HostOffloadPrf));

  if (hipMemcpy(HostOffloadPrf, DeviceOffloadPrf, sizeof(HostOffloadPrf),
                2 /*DToH*/) != 0) {
    PROF_ERR("%s\n", "Failed to copy offload prf structure from device");
    return -1;
  }

  void *DevCntsBegin = HostOffloadPrf[0];
  void *DevDataBegin = HostOffloadPrf[1];
  void *DevNamesBegin = HostOffloadPrf[2];
  void *DevUniformCntsBegin = HostOffloadPrf[3];
  void *DevCntsEnd = HostOffloadPrf[4];
  void *DevDataEnd = HostOffloadPrf[5];
  void *DevNamesEnd = HostOffloadPrf[6];
  void *DevUniformCntsEnd = HostOffloadPrf[7];

  if (IsVerboseMode()) {
    PROF_NOTE("%s", "Device Profile Pointers:\n");
    PROF_NOTE("  Counters:        %p - %p\n", DevCntsBegin, DevCntsEnd);
    PROF_NOTE("  Data:            %p - %p\n", DevDataBegin, DevDataEnd);
    PROF_NOTE("  Names:           %p - %p\n", DevNamesBegin, DevNamesEnd);
    PROF_NOTE("  UniformCounters: %p - %p\n", DevUniformCntsBegin,
              DevUniformCntsEnd);
  }

  size_t CountersSize = (char *)DevCntsEnd - (char *)DevCntsBegin;
  size_t DataSize = (char *)DevDataEnd - (char *)DevDataBegin;
  size_t NamesSize = (char *)DevNamesEnd - (char *)DevNamesBegin;
  size_t UniformCountersSize =
      (char *)DevUniformCntsEnd - (char *)DevUniformCntsBegin;

  if (IsVerboseMode()) {
    PROF_NOTE("Section sizes: Counters=%zu, Data=%zu, Names=%zu, "
              "UniformCounters=%zu\n",
              CountersSize, DataSize, NamesSize, UniformCountersSize);
  }

  if (CountersSize == 0 || DataSize == 0) {
    if (IsVerboseMode())
      PROF_NOTE("%s\n", "Counters or Data section has zero size. No profile "
                        "data to collect.");
    return 0;
  }

  // Pre-register device section symbols with CLR memory tracking.
  // This makes the section base pointers (and sub-pointers) safe for hipMemcpy.
  if (IsVerboseMode())
    PROF_NOTE("Pre-registering %d section symbols\n",
              NumSectionShadowVariables);
  for (int i = 0; i < NumSectionShadowVariables; ++i) {
    void *DevPtr = NULL;
    (void)hipGetSymbolAddress(&DevPtr, OffloadSectionShadowVariables[i]);
  }

  int ret = -1;

  // Allocate host memory for the device sections
  char *HostCountersBegin = (char *)malloc(CountersSize);
  char *HostDataBegin = (char *)malloc(DataSize);
  char *HostNamesBegin = (char *)malloc(NamesSize);
  char *HostUniformCountersBegin =
      (UniformCountersSize > 0) ? (char *)malloc(UniformCountersSize) : NULL;

  if (!HostCountersBegin || !HostDataBegin ||
      (NamesSize > 0 && !HostNamesBegin) ||
      (UniformCountersSize > 0 && !HostUniformCountersBegin)) {
    PROF_ERR("%s\n", "Failed to allocate host memory for device sections");
    goto cleanup;
  }

  // Copy data from device to host using HIP.
  if (memcpyDeviceToHost(HostCountersBegin, DevCntsBegin, CountersSize) != 0 ||
      memcpyDeviceToHost(HostDataBegin, DevDataBegin, DataSize) != 0 ||
      (NamesSize > 0 &&
       memcpyDeviceToHost(HostNamesBegin, DevNamesBegin, NamesSize) != 0) ||
      (UniformCountersSize > 0 &&
       memcpyDeviceToHost(HostUniformCountersBegin, DevUniformCntsBegin,
                          UniformCountersSize) != 0)) {
    PROF_ERR("%s\n", "Failed to copy profile sections from device");
    goto cleanup;
  }

  if (IsVerboseMode())
    PROF_NOTE("Copied device sections: Counters=%zu, Data=%zu, Names=%zu, "
              "UniformCounters=%zu\n",
              CountersSize, DataSize, NamesSize, UniformCountersSize);

  if (IsVerboseMode() && UniformCountersSize > 0) {
    PROF_NOTE("Successfully copied %zu bytes of uniform counters from device\n",
              UniformCountersSize);
  }

  // Compute padding sizes for proper buffer layout
  // lprofWriteDataImpl computes CountersDelta = CountersBegin - DataBegin
  // We need to arrange our buffer so this matches the expected file layout
  const uint64_t NumData = DataSize / sizeof(__llvm_profile_data);
  const uint64_t NumBitmapBytes = 0;
  const uint64_t VTableSectionSize = 0;
  const uint64_t VNamesSize = 0;
  uint64_t PaddingBytesBeforeCounters, PaddingBytesAfterCounters,
      PaddingBytesAfterBitmapBytes, PaddingBytesAfterNames,
      PaddingBytesAfterVTable, PaddingBytesAfterVNames;

  if (__llvm_profile_get_padding_sizes_for_counters(
          DataSize, CountersSize, NumBitmapBytes, NamesSize, VTableSectionSize,
          VNamesSize, &PaddingBytesBeforeCounters, &PaddingBytesAfterCounters,
          &PaddingBytesAfterBitmapBytes, &PaddingBytesAfterNames,
          &PaddingBytesAfterVTable, &PaddingBytesAfterVNames) != 0) {
    PROF_ERR("%s\n", "Failed to get padding sizes");
    goto cleanup;
  }

  // Create contiguous buffer with layout: [Data][Padding][Counters][Names]
  // This ensures CountersBegin - DataBegin = DataSize + PaddingBytesBeforeCounters
  size_t ContiguousBufferSize =
      DataSize + PaddingBytesBeforeCounters + CountersSize + NamesSize;
  char *ContiguousBuffer = (char *)malloc(ContiguousBufferSize);
  if (!ContiguousBuffer) {
    PROF_ERR("%s\n", "Failed to allocate contiguous buffer");
    goto cleanup;
  }
  memset(ContiguousBuffer, 0, ContiguousBufferSize);

  // Set up pointers into the contiguous buffer
  char *BufDataBegin = ContiguousBuffer;
  char *BufCountersBegin = ContiguousBuffer + DataSize + PaddingBytesBeforeCounters;
  char *BufNamesBegin = BufCountersBegin + CountersSize;

  // Copy data into contiguous buffer
  memcpy(BufDataBegin, HostDataBegin, DataSize);
  memcpy(BufCountersBegin, HostCountersBegin, CountersSize);
  memcpy(BufNamesBegin, HostNamesBegin, NamesSize);

  // Relocate CounterPtr in data records for file layout
  // CounterPtr is device-relative offset; we need to adjust for file layout
  // where Data section comes first, then Counters section
  __llvm_profile_data *RelocatedData = (__llvm_profile_data *)BufDataBegin;
  for (uint64_t i = 0; i < NumData; ++i) {
    if (RelocatedData[i].CounterPtr) {
      ptrdiff_t DeviceCounterPtrOffset = (ptrdiff_t)RelocatedData[i].CounterPtr;
      void *DeviceDataStructAddr =
          (char *)DevDataBegin + (i * sizeof(__llvm_profile_data));
      void *DeviceCountersAddr =
          (char *)DeviceDataStructAddr + DeviceCounterPtrOffset;
      ptrdiff_t OffsetIntoCountersSection =
          (char *)DeviceCountersAddr - (char *)DevCntsBegin;

      // New offset: from this data record to its counters in file layout
      // CountersDelta = BufCountersBegin - BufDataBegin = DataSize + Padding
      // CounterPtr = CountersDelta + OffsetIntoCounters - (i * sizeof)
      ptrdiff_t NewRelativeOffset = DataSize + PaddingBytesBeforeCounters +
                                    OffsetIntoCountersSection -
                                    (i * sizeof(__llvm_profile_data));
      memcpy(&RelocatedData[i].CounterPtr, &NewRelativeOffset,
             sizeof(NewRelativeOffset));
    }
    // Zero out unused fields
    memset(&RelocatedData[i].BitmapPtr, 0,
           sizeof(RelocatedData[i].BitmapPtr) +
               sizeof(RelocatedData[i].FunctionPointer) +
               sizeof(RelocatedData[i].Values));
  }

  // Build TU suffix string for filename
  char TUIndexStr[16] = "";
  if (TUIndex >= 0) {
    snprintf(TUIndexStr, sizeof(TUIndexStr), "%d", TUIndex);
  }

  // Use shared profile writing API
  const char *TargetTriple = "amdgcn-amd-amdhsa";
  ret = __llvm_write_custom_profile(
      TargetTriple, TUIndex >= 0 ? TUIndexStr : NULL,
      (__llvm_profile_data *)BufDataBegin,
      (__llvm_profile_data *)(BufDataBegin + DataSize), BufCountersBegin,
      BufCountersBegin + CountersSize,
      HostUniformCountersBegin,
      HostUniformCountersBegin
          ? HostUniformCountersBegin + UniformCountersSize
          : NULL,
      BufNamesBegin, BufNamesBegin + NamesSize, NULL);

  free(ContiguousBuffer);

  if (ret != 0) {
    PROF_ERR("%s\n", "Failed to write device profile using shared API");
  } else if (IsVerboseMode()) {
    PROF_NOTE("%s\n", "Successfully wrote device profile using shared API");
  }

cleanup:
  free(HostCountersBegin);
  free(HostDataBegin);
  free(HostNamesBegin);
  free(HostUniformCountersBegin);
  return ret;
}

static int ProcessShadowVariable(void *ShadowVar, int TUIndex) {
  void *DeviceOffloadPrf = NULL;
  if (hipGetSymbolAddress(&DeviceOffloadPrf, ShadowVar) != 0) {
    PROF_WARN("Failed to get symbol address for shadow variable %p\n",
              ShadowVar);
    return -1;
  }
  return ProcessDeviceOffloadPrf(DeviceOffloadPrf, TUIndex);
}

/* -------------------------------------------------------------------------- */
/*  Collect device-side profile data                                          */
/* -------------------------------------------------------------------------- */

int __llvm_profile_hip_collect_device_data(void) {
  if (IsVerboseMode())
    PROF_NOTE("%s", "__llvm_profile_hip_collect_device_data called\n");

  int Ret = 0;

  /* Shadow variables (static-linked kernels) */
  /* Always use TU index for consistent naming
   * (profile.amdgcn-amd-amdhsa.0.profraw, etc.) */
  for (int i = 0; i < NumShadowVariables; ++i) {
    if (ProcessShadowVariable(OffloadShadowVariables[i], i) != 0)
      Ret = -1;
  }

  /* Dynamically-loaded modules */
  for (int i = 0; i < NumDynamicModules; ++i) {
    OffloadDynamicModuleInfo *Info = &DynamicModules[i];
    if (!Info->Processed) {
      PROF_WARN("Dynamic module %p was not processed before unload\n",
                Info->ModulePtr);
      Ret = -1;
    }
  }

  return Ret;
}

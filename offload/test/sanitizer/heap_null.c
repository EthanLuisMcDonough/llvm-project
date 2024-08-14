// clang-format off
// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

// Align lines.

#include <stdint.h>
#include <stdio.h>

int main(void) {

  int *Null = 0;
#pragma omp target
  {
    // clang-format off
    // CHECK:      ERROR: OffloadSanitizer out-of-bounds access on address 0x0000000000000000 at pc [[PC:0x.*]]
    // CHECK-NEXT: WRITE of size 4 at 0x0000000000000000 thread <0, 0, 0> block <0, 0, 0>
    // CHECK-NEXT: #0 [[PC]] main null.c:[[@LINE+3]]
    // CHECK-NEXT: 0x0000000000000000 is located 0 bytes inside of 0-byte region [0x0000000000000000,0x0000000000000000)
    // clang-format on
    *Null = 42;
  }
}

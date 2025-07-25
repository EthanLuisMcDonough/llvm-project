; NOTE: Assertions have been autogenerated by utils/update_mir_test_checks.py UTC_ARGS: --version 4
; RUN: llc -global-isel -mtriple=amdgcn -stop-after=irtranslator < %s | FileCheck %s

define void @prefetch_read(ptr %ptr) {
  ; CHECK-LABEL: name: prefetch_read
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   liveins: $vgpr0, $vgpr1
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
  ; CHECK-NEXT:   [[COPY1:%[0-9]+]]:_(s32) = COPY $vgpr1
  ; CHECK-NEXT:   [[MV:%[0-9]+]]:_(p0) = G_MERGE_VALUES [[COPY]](s32), [[COPY1]](s32)
  ; CHECK-NEXT:   G_PREFETCH [[MV]](p0), 0, 0, 0 :: (load unknown-size from %ir.ptr, align 1)
  ; CHECK-NEXT:   SI_RETURN
  call void @llvm.prefetch.p0(ptr %ptr, i32 0, i32 0, i32 0)
  ret void
}

define void @prefetch_write(ptr %ptr) {
  ; CHECK-LABEL: name: prefetch_write
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   liveins: $vgpr0, $vgpr1
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:_(s32) = COPY $vgpr0
  ; CHECK-NEXT:   [[COPY1:%[0-9]+]]:_(s32) = COPY $vgpr1
  ; CHECK-NEXT:   [[MV:%[0-9]+]]:_(p0) = G_MERGE_VALUES [[COPY]](s32), [[COPY1]](s32)
  ; CHECK-NEXT:   G_PREFETCH [[MV]](p0), 1, 1, 1 :: (store unknown-size into %ir.ptr, align 1)
  ; CHECK-NEXT:   SI_RETURN
  call void @llvm.prefetch.p0(ptr %ptr, i32 1, i32 1, i32 1)
  ret void
}

declare void @llvm.prefetch.p0(ptr, i32, i32, i32)

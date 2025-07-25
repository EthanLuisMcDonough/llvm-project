; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; Checks that move-auto-init can move instruction passed unclobbering memory
; instructions.
; RUN: opt < %s -S -passes='move-auto-init' -verify-memoryssa | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2) #0 {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    [[TMP4:%.*]] = alloca [100 x i8], align 16
; CHECK-NEXT:    [[TMP5:%.*]] = alloca [2 x i8], align 1
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds [100 x i8], ptr [[TMP4]], i64 0, i64 0
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 100, ptr nonnull [[TMP4]]) #[[ATTR3:[0-9]+]]
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds [2 x i8], ptr [[TMP5]], i64 0, i64 0
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[TMP5]]) #[[ATTR3]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds [2 x i8], ptr [[TMP5]], i64 0, i64 1
; CHECK-NEXT:    [[TMP9:%.*]] = icmp eq i32 [[TMP1:%.*]], 0
; CHECK-NEXT:    br i1 [[TMP9]], label [[TMP15:%.*]], label [[TMP10:%.*]]
; CHECK:       10:
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(100) [[TMP6]], i8 -86, i64 100, i1 false), !annotation [[META0:![0-9]+]]
; CHECK-NEXT:    [[TMP11:%.*]] = sext i32 [[TMP0:%.*]] to i64
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds [100 x i8], ptr [[TMP4]], i64 0, i64 [[TMP11]]
; CHECK-NEXT:    store i8 12, ptr [[TMP12]], align 1
; CHECK-NEXT:    [[TMP13:%.*]] = load i8, ptr [[TMP6]], align 16
; CHECK-NEXT:    [[TMP14:%.*]] = sext i8 [[TMP13]] to i32
; CHECK-NEXT:    br label [[TMP22:%.*]]
; CHECK:       15:
; CHECK-NEXT:    [[TMP16:%.*]] = icmp eq i32 [[TMP2:%.*]], 0
; CHECK-NEXT:    br i1 [[TMP16]], label [[TMP22]], label [[TMP17:%.*]]
; CHECK:       17:
; CHECK-NEXT:    store i8 -86, ptr [[TMP7]], align 1, !annotation [[META0]]
; CHECK-NEXT:    store i8 -86, ptr [[TMP8]], align 1, !annotation [[META0]]
; CHECK-NEXT:    [[TMP18:%.*]] = sext i32 [[TMP0]] to i64
; CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds [2 x i8], ptr [[TMP5]], i64 0, i64 [[TMP18]]
; CHECK-NEXT:    store i8 12, ptr [[TMP19]], align 1
; CHECK-NEXT:    [[TMP20:%.*]] = load i8, ptr [[TMP7]], align 1
; CHECK-NEXT:    [[TMP21:%.*]] = sext i8 [[TMP20]] to i32
; CHECK-NEXT:    br label [[TMP22]]
; CHECK:       22:
; CHECK-NEXT:    [[TMP23:%.*]] = phi i32 [ [[TMP14]], [[TMP10]] ], [ [[TMP21]], [[TMP17]] ], [ 0, [[TMP15]] ]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[TMP5]]) #[[ATTR3]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 100, ptr nonnull [[TMP4]]) #[[ATTR3]]
; CHECK-NEXT:    ret i32 [[TMP23]]
;

  %4 = alloca [100 x i8], align 16
  %5 = alloca [2 x i8], align 1
  %6 = getelementptr inbounds [100 x i8], ptr %4, i64 0, i64 0
  call void @llvm.lifetime.start.p0(i64 100, ptr nonnull %4) #3
  ; This memset must move.
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(100) %6, i8 -86, i64 100, i1 false), !annotation !0
  %7 = getelementptr inbounds [2 x i8], ptr %5, i64 0, i64 0
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %5) #3
  ; This store must move.
  store i8 -86, ptr %7, align 1, !annotation !0
  %8 = getelementptr inbounds [2 x i8], ptr %5, i64 0, i64 1
  ; This store must move.
  store i8 -86, ptr %8, align 1, !annotation !0
  %9 = icmp eq i32 %1, 0
  br i1 %9, label %15, label %10

10:
  %11 = sext i32 %0 to i64
  %12 = getelementptr inbounds [100 x i8], ptr %4, i64 0, i64 %11
  store i8 12, ptr %12, align 1
  %13 = load i8, ptr %6, align 16
  %14 = sext i8 %13 to i32
  br label %22

15:
  %16 = icmp eq i32 %2, 0
  br i1 %16, label %22, label %17

17:
  %18 = sext i32 %0 to i64
  %19 = getelementptr inbounds [2 x i8], ptr %5, i64 0, i64 %18
  store i8 12, ptr %19, align 1
  %20 = load i8, ptr %7, align 1
  %21 = sext i8 %20 to i32
  br label %22

22:
  %23 = phi i32 [ %14, %10 ], [ %21, %17 ], [ 0, %15 ]
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(i64 100, ptr nonnull %4) #3
  ret i32 %23
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #2

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { mustprogress nofree nosync nounwind readnone uwtable willreturn }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { argmemonly mustprogress nofree nounwind willreturn writeonly }
attributes #3 = { nounwind }

!0 = !{!"auto-init"}

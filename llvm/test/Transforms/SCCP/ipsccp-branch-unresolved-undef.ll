; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature
; RUN: opt < %s -S -passes=ipsccp | FileCheck %s
;; Check that @patatino is optimised to "unreachable" given that it branches on
;; undef. Check too that debug intrinsics have no effect on this.

define void @main() {
; CHECK-LABEL: define {{[^@]+}}@main() {
; CHECK-NEXT:    [[CALL:%.*]] = call i1 @patatino(i1 undef)
; CHECK-NEXT:    ret void
;
  %call = call i1 @patatino(i1 undef)
  ret void
}

define internal i1 @patatino(i1 %a) {
; CHECK-LABEL: define {{[^@]+}}@patatino
; CHECK-SAME: (i1 [[A:%.*]]) {
; CHECK-NEXT:    unreachable
;
  br i1 %a, label %ontrue, label %onfalse
ontrue:
  call void @llvm.dbg.value(metadata i32 0, metadata !6, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !9, metadata !DIExpression()), !dbg !11
  ret i1 false
onfalse:
  call void @llvm.dbg.value(metadata i32 0, metadata !6, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 1, metadata !9, metadata !DIExpression()), !dbg !11
  ret i1 false
}

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!21}
!llvm.dbg.cu = !{!2}

!0 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !20, scope: !1, type: !3)
!1 = !DIFile(filename: "b.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang", isOptimized: true, emissionKind: FullDebug, file: !20)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "i", line: 2, arg: 1, scope: !0, file: !1, type: !5)
!7 = !DILocation(line: 2, column: 13, scope: !0)
!9 = !DILocalVariable(name: "k", line: 3, scope: !10, file: !1, type: !5)
!10 = distinct !DILexicalBlock(line: 2, column: 16, file: !20, scope: !0)
!11 = !DILocation(line: 3, column: 12, scope: !10)
!12 = !DILocation(line: 4, column: 3, scope: !10)
!13 = !DILocation(line: 5, column: 5, scope: !14)
!14 = distinct !DILexicalBlock(line: 4, column: 10, file: !20, scope: !10)
!15 = !DILocation(line: 6, column: 3, scope: !14)
!16 = !DILocation(line: 7, column: 5, scope: !17)
!17 = distinct !DILexicalBlock(line: 6, column: 10, file: !20, scope: !10)
!18 = !DILocation(line: 8, column: 3, scope: !17)
!19 = !DILocation(line: 9, column: 3, scope: !10)
!20 = !DIFile(filename: "b.c", directory: "/private/tmp")
!21 = !{i32 1, !"Debug Info Version", i32 3}


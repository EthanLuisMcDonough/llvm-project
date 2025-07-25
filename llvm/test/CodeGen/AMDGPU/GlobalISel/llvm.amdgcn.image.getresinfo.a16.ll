; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1100 -mattr=-real-true16 < %s | FileCheck -check-prefixes=GFX10 %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1200 -mattr=-real-true16 < %s | FileCheck -check-prefixes=GFX12 %s

define amdgpu_ps <4 x float> @getresinfo_1d(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_1d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_1d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_1d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_2d(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_2d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_2d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_2d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2d.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_3d(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_3d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_3d:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_3d:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.3d.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_cube(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_cube:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_cube:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_cube:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.cube.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_1darray(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_1darray:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_1darray:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_1darray:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.1darray.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_2darray(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_2darray:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_2darray:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_2darray:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2darray.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_2dmsaa(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_2dmsaa:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_2dmsaa:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_2dmsaa:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2dmsaa.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_2darraymsaa(<8 x i32> inreg %rsrc, i16 %mip) {
; GFX9-LABEL: getresinfo_2darraymsaa:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    s_mov_b32 s0, s2
; GFX9-NEXT:    s_mov_b32 s1, s3
; GFX9-NEXT:    s_mov_b32 s2, s4
; GFX9-NEXT:    s_mov_b32 s3, s5
; GFX9-NEXT:    s_mov_b32 s4, s6
; GFX9-NEXT:    s_mov_b32 s5, s7
; GFX9-NEXT:    s_mov_b32 s6, s8
; GFX9-NEXT:    s_mov_b32 s7, s9
; GFX9-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf unorm a16 da
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_2darraymsaa:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    s_mov_b32 s0, s2
; GFX10-NEXT:    s_mov_b32 s1, s3
; GFX10-NEXT:    s_mov_b32 s2, s4
; GFX10-NEXT:    s_mov_b32 s3, s5
; GFX10-NEXT:    s_mov_b32 s4, s6
; GFX10-NEXT:    s_mov_b32 s5, s7
; GFX10-NEXT:    s_mov_b32 s6, s8
; GFX10-NEXT:    s_mov_b32 s7, s9
; GFX10-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm a16
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_2darraymsaa:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    s_mov_b32 s0, s2
; GFX12-NEXT:    s_mov_b32 s1, s3
; GFX12-NEXT:    s_mov_b32 s2, s4
; GFX12-NEXT:    s_mov_b32 s3, s5
; GFX12-NEXT:    s_mov_b32 s4, s6
; GFX12-NEXT:    s_mov_b32 s5, s7
; GFX12-NEXT:    s_mov_b32 s6, s8
; GFX12-NEXT:    s_mov_b32 s7, s9
; GFX12-NEXT:    image_get_resinfo v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY a16
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.2darraymsaa.v4f32.i16(i32 15, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

define amdgpu_ps <4 x float> @getresinfo_dmask0(<8 x i32> inreg %rsrc, <4 x float> %vdata, i16 %mip) {
; GFX9-LABEL: getresinfo_dmask0:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: getresinfo_dmask0:
; GFX10:       ; %bb.0: ; %main_body
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX12-LABEL: getresinfo_dmask0:
; GFX12:       ; %bb.0: ; %main_body
; GFX12-NEXT:    ; return to shader part epilog
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i16(i32 0, i16 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %r
}

declare <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.2d.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.3d.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.cube.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.1darray.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.2darray.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.2dmsaa.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1
declare <4 x float> @llvm.amdgcn.image.getresinfo.2darraymsaa.v4f32.i16(i32 immarg, i16, <8 x i32>, i32 immarg, i32 immarg) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

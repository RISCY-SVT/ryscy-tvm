	.text
	.attribute	4, 16
	.attribute	5, "rv64gcv0p7"
	.file	"autovec.c"
	.globl	autovec_add
	.p2align	1
	.type	autovec_add,@function
autovec_add:
	beqz	a3, .LBB0_6
	csrr	t0, vlenb
	slli	t1, t0, 1
	bltu	a3, t1, .LBB0_3
	sub	a4, a2, a0
	sltu	a4, a4, t1
	sub	a5, a2, a1
	sltu	a5, a5, t1
	or	a4, a4, a5
	beqz	a4, .LBB0_7
.LBB0_3:
	li	a6, 0
.LBB0_4:
	sub	a3, a3, a6
	add	a2, a2, a6
	add	a1, a1, a6
	add	a0, a0, a6
.LBB0_5:
	lb	a4, 0(a0)
	lb	a5, 0(a1)
	add	a4, a4, a5
	sb	a4, 0(a2)
	addi	a3, a3, -1
	addi	a2, a2, 1
	addi	a1, a1, 1
	addi	a0, a0, 1
	bnez	a3, .LBB0_5
.LBB0_6:
	ret
.LBB0_7:
	addi	a4, t1, -1
	and	a7, a3, a4
	sub	a6, a3, a7
	vsetvli	a4, zero, e8, m1
	mv	t2, a6
	mv	t3, a2
	mv	t4, a1
	mv	a5, a0
.LBB0_8:
	vl1r.v	v8, (a5)
	add	a4, a5, t0
	vl1r.v	v9, (a4)
	vl1r.v	v10, (t4)
	add	a4, t4, t0
	vl1r.v	v11, (a4)
	vadd.vv	v8, v10, v8
	vadd.vv	v9, v11, v9
	vs1r.v	v8, (t3)
	add	a4, t3, t0
	vs1r.v	v9, (a4)
	add	a5, a5, t1
	add	t4, t4, t1
	sub	t2, t2, t1
	add	t3, t3, t1
	bnez	t2, .LBB0_8
	bnez	a7, .LBB0_4
	j	.LBB0_6
.Lfunc_end0:
	.size	autovec_add, .Lfunc_end0-autovec_add

	.globl	main
	.p2align	1
	.type	main,@function
main:
	li	a0, 0
	ret
.Lfunc_end1:
	.size	main, .Lfunc_end1-main

	.ident	"clang version 16.0.0 (https://github.com/dkurt/llvm-rvv-071 b027aa1b59c9f53240bdc836f39656723fdf9df0)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

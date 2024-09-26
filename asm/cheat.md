## レジスタ
- rax (64bit), eax (32bit), ax (16bit), al (8bit). 低ビットのものは, raxと同じ箱の下位ビットを指している.
- `movsx rax, al` は, alの符号拡張をして, raxに入れる. movzxは符号なしの場合に使う.
- `movsxd rax, eax` は32(dward)から64bit(qward)に拡張する. 32bitの値が符号付きの場合に使う.

## addressing
- `lea r11 [SomeData]` は, r11にSomeDataのアドレスを入れる. つまり, r11 = &SomeData

## 整数演算
- `mul rsi` は `rax = rax * rsi` と同じ

## 関数の引数の渡し方 (nasm)
- 最初の6変数はrdi, rsi, rdx, rcx, r8, r9 レジスタに渡す
- 覚え方: Dizzy Dixie 89 https://stackoverflow.com/questions/63891991/whats-the-best-way-to-remember-the-x86-64-system-v-arg-register-order
- 7番目以降はスタックに積み上げる. e.g. 7番目の引数は`[rsp+8], 8番目の引数は[rsp+16]`となる
- 戻り値はraxレジスタに入れておいて, `ret` で返す



## Motions in Microseconds via Vectorized Sampling-Based Planning
- 疑問: SIMDで高速化できてもオーダは10倍程度. どうやって500倍もの高速化を達成したのか?
- ベクトル化によって高速化
- conditional branchingを避けるように事前計算する.
    - FKについてはtracing compilarを使う
    - 条件分岐があるとパイプライン化が難しくなってしまう.
    - 条件分岐があるとloop unrolling(ループ展開)が難しくなってしまう.
        - ループ展開できれば, ループの条件チェックしなくてすむし, pipeline化もしやすい.
        - もちろんSIMD化もしやすい.
- array of structures (AoS) から structure of arrays (SoA) に変換することで高速化

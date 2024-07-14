## Bridge Datasetを使ったpretrained model + 10回くらいのfine tuning RL [Kumar+, 2022]
- データセット
    - 中身は D={(s, a, s+, r)}であり, Bridge Dataset https://rail-berkeley.github.io/bridgedata/ 自体は (s, a, s+)だけをもっている.
    - rは最後の3stepのみ+1の報酬, それ意外は0 
- fine-tune RLにおけるrewardの定義
    - image classifier による方法 [Singh+, 2019]

* Kumar, Aviral, et al. "Pre-training for robots: Offline rl enables learning new tasks from a handful of trials." arXiv preprint arXiv:2210.05178 (2022).

## reference
- Kumar, Aviral, et al. "Pre-training for robots: Offline rl enables learning new tasks from a handful of trials." arXiv preprint arXiv:2210.05178 (2022).
- Singh, Avi, et al. "End-to-end robotic reinforcement learning without reward engineering." RSS 2019 (2019).

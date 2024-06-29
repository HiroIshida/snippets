key: classical planning

```bib
@article{chakraborti2017plan,
  title={Plan explanations as model reconciliation: Moving beyond explanation as soliloquy},
  author={Chakraborti, Tathagata and Sreedharan, Sarath and Zhang, Yu and Kambhampati, Subbarao},
  journal={arXiv preprint arXiv:1701.08317},
  year={2017}
}
```

## Model reconcilation の動機
ロボットが内部的にもつ計画モデル$M_R$と, 人間が内部的にもつ計画モデルを$M_H$とする.
ロボットの計画結果を人間に説明したいとする. 
このとき, ロボットが計画理由を独り言のようにぶつぶついうことはできるけど, それが人間には理解できない可能性が高い. なぜなら$M_R$と$M_H$が異なるからである.
そこで, その差異を先に人間に説明したい.
この差異は, 例えば, precondやゴール状態の条件に何が足りていないかなど.

## Model reconcilation の問題設定
最適経路$\pi^*$がロボットから与えられたとする.
あるモデル$M$に基づいてこの最適経路を解釈した際のコストを$C(\pi^*, M)$とする.
また, あるモデル$M$に基づいて計画を行った際の最適コストを$C_M^*$とする.
すなわち常に$C(\pi^*, M_R) = C_{M_R}^*$ではある. 
しかし, $M_H$で解釈すると$C(\pi^*, M_H) > C_{M_H}^*$の可能性がある.

モデルを一つの状態/集合として扱うためにモデルを集合に変換する関数$\Gamma$を考える.
あるモデル$M$とモデル$M'$の距離を$\Gamma(M)\Delta \Gamma(M')$の要素数で表す. ここで$\Delta$はsymmetric differenceのことである. Explanation $\epsilon$は次の条件を満たす量(logical statementの集合)である:

1. $\hat{M}_R = M_R + \epsilon$
2. $C(\pi^*, \hat{M}_R) = C^*_{\hat{M}_R}$
3. $C(\pi^*, M_R) = C^*_{M_R}$
すなわち, $\pi^*$を$M_R$と$\hat{M}_R$のどちらにおいても正しく説明するような変更量のことである.
model reconcilationはこの変更量を求めることに相当し, さらに, できれば小さな変更量を求める問題のことである.


## 分かりづらいポイント
1. $\pi^*$がロボットのだした計画であり, givenなものが書いていない. 本来なら$\pi^*_R$みたいに書くべき.
2. explanationの定義が書かれていない. 以下のjournalバージョンには書かれていた.
```bib
@article{sreedharan2021foundations,
  title={Foundations of explanations as model reconciliation},
  author={Sreedharan, Sarath and Chakraborti, Tathagata and Kambhampati, Subbarao},
  journal={Artificial Intelligence},
  volume={301},
  pages={103558},
  year={2021},
  publisher={Elsevier}
}
```
3. $\pi$^*は常に存在? そもその解がある問題を解くことを前提としている.

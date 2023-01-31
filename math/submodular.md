# maximum coverage problemにおけるgreedyが最適でない例
https://en.wikipedia.org/wiki/Maximum_coverage_problem
要素制約$k < 2$があるとする.
$s_1 = [0, 10], s_2 = [10, 13], s_3 = [5, 12], s_4 = [-2, 5]$の4つの集合があるとする.
greedy algorithmだと$s_1$, $s_2$の順に選ぶのでスコアは12.
最適化ケースは$s_3$, $s_4$の順に選ぶのでスコアは13.

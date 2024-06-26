# contact sequence motion joint optimization
## 代表的
- direct method / mode invariant optimization [Posa+, IJRR 2014] (696 citations)
    - https://journals.sagepub.com/doi/full/10.1177/0278364913506757
    - DDPのようなshooting法は, 少しのuの変化が大きなxの変化を引き起こすため最適化が難しい? 直接法ならこうした問題がおきない.
    - Linear complementarity problem (LCP)を解くことで, 接触力を考慮した最適化を行う.
    - complementarity constraints を考慮した非線形最適化問題をMathematical Program with Complementarity Constraints (MPCC)と呼ぶ.
    - MPCCはSNOPTで解ける.

- mode invariant optimization with kinematic constraints + centroidal dynamics [Dai+, Humanoids 2014] ($73 citations)
    - https://ieeexplore.ieee.org/abstract/document/7041375
    - 上のPosaの手法をkineamtic constraintsを考慮して拡張したもの.
    - dynamicsはcentroidal dynamicsを考慮している.
    - centroidal angular momentumは[Orin+, Auton. Robots 2013]式(19)を参照している.

- contact invariant optimization [Mordatch+, ACM ToG 2012] (564 citations)
    - https://dl.acm.org/doi/abs/10.1145/2185520.2185539

- phase-based end effector parameterization [Winkler+, RA-L 2018] (457 citations)
    - https://ieeexplore.ieee.org/abstract/document/8283570
    - 各足の接触状態がswing-stanceを繰り返すことに注目. 事前にフェーズを決めておくことができる!!
    - \Delta Tは複数個用意しておいて, そんなにたくさんのフェーズが必要ないなら, ただ\Delta T= 0とすればいいだけ.
    - 個人的に, この方法がいいとおもうのは, わかりやすさと汎用的なNLPが使えること. (SNOPTでもIpoptでも最適化できた.)

- Mixed-integer optimization [Aceituno-Cabezas+, RA-L 2017] (169 citations)
    - https://ieeexplore.ieee.org/abstract/document/8141917

## 最近のもの
- contact-implicit model predictive control [Le Cleac'h+, T-RO 2024] (52 citations)
    - https://ieeexplore.ieee.org/abstract/document/10384795
    - githubにソースコード公開されている. juliaだけど. https://github.com/dojo-sim/ContactImplicitMPC.jl
    - いろんな事前計算とコンタクトを扱うことに特化した数理最適化手法を開発してるっぽい.

- Winkler+のinitializationを学習 [Melon+, ICRA 2020] (34 citations)
    - https://ieeexplore.ieee.org/abstract/document/9196562
    - Table1をみると, 210msで最適化が終わってる. キネマティクス考慮してるのに, これってかなり遅い...?

# エンドエフェクタの力の釣り合いについて
- EFを多角形で表現し, その頂点において力の釣り合いを考慮する. These contact surfaces can generate unilateral contact forces...at their vertices [Kumagai+, RA-L 2021]
- 実際には力は足裏に分布してる. rotational motionを考える場合には, 各点における速度(というか進行方向)が異なり, これの積分値が摩擦力となる. なので, 足裏に多数のcontact pointsをとる. [Kojima+, RA-L 2017]

# centroidal dynamics
- 室岡さん[Murooka+, RA-L 2022] はfixed Inertia matrixを使っている. 

# citation
- Posa, Michael, Cecilia Cantu, and Russ Tedrake. "A direct method for trajectory optimization of rigid bodies through contact." The International Journal of Robotics Research 33.1 (2014): 69-81.
- Dai, Hongkai, Andrés Valenzuela, and Russ Tedrake. "Whole-body motion planning with centroidal dynamics and full kinematics." 2014 IEEE-RAS International Conference on Humanoid Robots. IEEE, 2014.
- Mordatch, Igor, Emanuel Todorov, and Zoran Popović. "Discovery of complex behaviors through contact-invariant optimization." ACM Transactions on Graphics (ToG) 31.4 (2012): 1-8.
- Winkler, Alexander W., et al. "Gait and trajectory optimization for legged systems through phase-based end-effector parameterization." IEEE Robotics and Automation Letters 3.3 (2018): 1560-1567.
- Aceituno-Cabezas, Bernardo, et al. "Simultaneous contact, gait, and motion planning for robust multilegged locomotion via mixed-integer convex optimization." IEEE Robotics and Automation Letters 3.3 (2017): 2531-2538.
- Le Cleac'h, Simon, et al. "Fast contact-implicit model predictive control." IEEE Transactions on Robotics (2024).
- Melon, Oliwier, et al. "Reliable trajectories for dynamic quadrupeds using analytical costs and learned initializations." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.
- Kumagai, Iori, et al. "Multi-contact locomotion planning with bilateral contact forces considering kinematics and statics during contact transition." IEEE Robotics and Automation Letters 6.4 (2021): 6654-6661.
- Kojima, Kunio, et al. "Rotational sliding motion generation for humanoid robot by force distribution in each contact face." IEEE Robotics and Automation Letters 2.4 (2017): 2088-2095.
- Orin, David E., Ambarish Goswami, and Sung-Hee Lee. "Centroidal dynamics of a humanoid robot." Autonomous robots 35 (2013): 161-176.
- Murooka, Masaki, Mitsuharu Morisawa, and Fumio Kanehiro. "Centroidal trajectory generation and stabilization based on preview control for humanoid multi-contact motion." IEEE Robotics and Automation Letters 7.3 (2022): 8225-8232.

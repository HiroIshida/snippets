# 前提
skmp == 0.0.1 を使用.

# pr2をpybullet上で動かす場合の注意点
- ["l_gripper_motor_screw_joint", "r_gripper_motor_screw_joint"] は振動するので無視する.
- gripperを動かす場合には4つのジョイントを動かす必要あり.
- gripperのjointに対してsetJointMotorControl2しても, 全然追従しない. 小刻みにangleを変化させながらresetJointStateをいれていくのがよい.
- forceはある程度大きくしないとgraspできない.
- 大きくしすぎると振動してしまう. ので, defaultでforce=500として, torsoだけ2000とかにする. 
- torsoは荷重を支えるために大きなforceが必要.
- defaultのdtを使わない.

# pybulletのしくみ
クライアントは: まずpybullet.c内で, コマンドを生成する(b3CalculateInverseKinematicsCommandInit). . 

```c++
// inside PhysicsClientC_API.cpp
B3_SHARED_API b3SharedMemoryCommandHandle b3CalculateInverseKinematicsCommandInit(b3PhysicsClientHandle physClient, int bodyUniqueId){
	PhysicsClient* cl = (PhysicsClient*)physClient;
	struct SharedMemoryCommand* command = cl->getAvailableSharedMemoryCommand();
	command->m_type = CMD_CALCULATE_INVERSE_KINEMATICS;
	command->m_calculateInverseKinematicsArguments.m_bodyUniqueId = bodyUniqueId;
	return (b3SharedMemoryCommandHandle)command;
}
```
それぞれのコマンドは, コマンド識別用の`m_type`というメンバに加えて, その他の処理部にわたすための引数をまとめた構造体(例えば: を内部にもっている. その定義は, こんな感じ. command init関数内では上記のhogehogeArgsのメンバ変数を埋めていく操作をしていることに注意.  
```c++
// inside SharedMemoryCommands.h
struct CalculateInverseKinematicsResultArgs{
	int m_bodyUniqueId;
	int m_dofCount;
	double m_jointPositions[MAX_DEGREE_OF_FREEDOM];
};
// blah blah 
struct CalculateInverseKinematicsArgs m_calculateInverseKinematicsArguments;
```
こうしてつくられたコマンドを`b3SubmitClientCommandAndWaitStatus`によってサーバに送りつける. サーバ`PhysicsServerCommandProcessor.cpp`内のswitch文によって, dispatchされて最終的にメイン処理関数に到達する. メイン処理関数はぜんぶ次の形式:  
```c++
bool PhysicsServerCommandProcessor::processCalculateInverseKinematicsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes)
```
大事なのは, `clientCmd`と`serverStatusOut`である. 前者は先述した, クライアントから届いたコマンドである. 後者は, 処理に成功したかどうかの情報と実行結果データを蓄えたものである. 成功の可否はこんな感じでセットされる: 
```
serverCmd.m_type = CMD_CALCULATE_INVERSE_KINEMATICS_COMPLETED;
serverCmd.m_type = CMD_CALCULATE_INVERSE_KINEMATICS_FAILED; 
```
物理エンジンをダイレクトに使う場合とGUI(shared memory?)を用いる場合それぞれ, 
```c++
void PhysicsDirect::postProcessStatus(const struct SharedMemoryStatus& serverCmd)
const SharedMemoryStatus* PhysicsClientSharedMemory::processServerStatus()
```
のような感じでチェックされるようなので, 新しいメソッドを追加する場合には両方にcaseを追加することをお忘れなく. このチェックが終わったあとは, `b3SubmitClientCommandAndWaitStatus`の結果としてstatusのハンドルが返ってくるので, この内部変数を取り出すことで, 所望のデータを得ることができる. clientの時と同じようにサーバが返してくる(sever)statusも内部にデータ保存用構造体をもっている. たとえば: 
```
// inside SharedMemoryCommands.h
struct CalculateInverseKinematicsResultArgs
{
	int m_bodyUniqueId;
	int m_dofCount;
	double m_jointPositions[MAX_DEGREE_OF_FREEDOM];
};
// blah blah 
struct CalculateInverseKinematicsResultArgs m_inverseKinematicsResultArgs;
```
のような感じ. 最後に, `CMD_HOGE_HOGE`のようなものはすべて`SharedMemoryPubulic.h`内でenumで定義されていることに注意.  



# pitfalls
## when joint limit is not specified in urdf
lower and upper limits are set to 0.0 and -1.0. As the bullet (not pybullet) documentation shows, if lower > upper, that means the joint is free.

## calculateJacobian
Please specify that the base is fixed:
```python
robot = pb.loadURDF("./robot/config.urdf", useFixedBase=True)
```
otherwise the `calcualateJacobian` will return 3x(n+6) matrix. The additional 6 dims are for xyzrpy.

## getCameraImage
`bullet3/examples/SharedMemory/plugins/tinyRendererPlugin`
`bullet3/examples/OpenGLWindow`

## about rayTest
Declaration:
```cpp
virtual void rayTest(const btVector3& rayFromWorld, const btVector3& rayToWorld, RayResultCallback& resultCallback) const;
```
First by using a aabb collision checking (collision of aligned boxes) which implemented in `BroadPhaseCollision`, then perform precise collision checking by `rayTestSingleInternal`, where different processes are used for convex and non-convex object cases.
```cpp
void btCollisionWorld::rayTestSingleInternal(const btTransform& rayFromTrans, const btTransform& rayToTrans,
											 const btCollisionObjectWrapper* collisionObjectWrap,
											 RayResultCallback& resultCallback)
```

## how inverse kinematics method is imported into python
1. `BussIK` defined main method. `Jacobian::CalcDeltaThetas*` compute `dq` and set it to a public property `dTheta`.
2. `IKTrajectoryHelper.h` wraps `BussIK`. `computeIK` method call `CalcDeltaThetas*` mentioned above and set `q_new = q_current + dq` (`*q_new` is returned as a argument).
3. `PhysicsServerCommandProcessor` calls `computeIK` inside `processCalculateInverseKinematicsCommand`.

defined in `BussIkjk` then wrapped by `IKTrajectoryHelper.h`. This header is included by `PhysicsServerCommandProcessor.cpp`. Then finally integrated into `*C_API`.

## calcualateJacobians 
Super confusing thing is, `calcualateJacobians` method seem to come from `src/BulletInverseDynamics/MultBodyTree.hpp`. Insider `PhysicsServerCommandProcessor::processCalculateJacobianCommand`,  `tree->getBodyJacobianTrans` and `tree->getBodyJacobianRot` are called and computed values are stored as `jac_t` and `jac_r`. Seems that `jac_r` is used just to compute `jac_t_new`, and not returned.


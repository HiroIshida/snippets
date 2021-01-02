以下のコードで, `for joint in tree_joints(state.mechanism)` とあるが, treeを生成するさいに, 根本から順番にedgeを足していっているので, edgeはその順番になっている. これによって, `transforms_to_root[parentid]` が常にdirtyではない状態になる. ただ, この方法だと一部のジョイントを動かしただけで全体のTFがアップデートされるので非効率である. 実際tinyfkで0.7秒で解けるFKが1.5秒かかってしまっている.  
```julia
# Cache variable update functions
@inline update_transforms!(state::MechanismState) = isdirty(state.transforms_to_root) && _update_transforms!(state)
@noinline function _update_transforms!(state::MechanismState)
    @modcountcheck state state.mechanism

    # update tree joint transforms
    qs = values(segments(state.q))
    @inbounds map!(joint_transform, state.tree_joint_transforms, state.treejoints, qs)

    # update transforms to root
    transforms_to_root = state.transforms_to_root
    for joint in tree_joints(state.mechanism)
        jointid = JointID(joint)
        parentid, bodyid = predsucc(jointid, state)
        transforms_to_root[bodyid] = transforms_to_root[parentid] * joint_to_predecessor(joint) * state.joint_transforms[jointid]
    end
    state.transforms_to_root.dirty = false

    # update non-tree joint transforms
    if !isempty(state.nontreejointids)
        broadcast!(state.non_tree_joint_transforms, state, state.nontreejoints) do state, joint
            predid, succid = predsucc(JointID(joint), state)
            before_to_root = state.transforms_to_root[predid] * joint_to_predecessor(joint)
            after_to_root = state.transforms_to_root[succid] * joint_to_successor(joint)
            inv(before_to_root) * after_to_root
        end
    end
    state.joint_transforms.dirty = false
    nothing
end
```

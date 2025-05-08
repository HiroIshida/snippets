(define (stream tamp) 

  (:stream sample-pose
           :inputs (?object)
           :domain (IsGraspable ?object)
           :outputs (?pose)
           :certified (TypePose ?pose)
  )

  (:stream sample-grasp
           :inputs (?object)
           :domain (IsGraspable ?object)
           :outputs (?grasp)
           :certified (and (IsGrasp ?grasp ?object) (TypeGrasp ?grasp))
  )
  (:stream sample-path-to-grasp ;; assuming that plan starts from the home configuration
           :inputs (?object ?pose ?grasp)
           :domain (and
                     (IsGraspable ?object)
                     (TypePose ?pose)
                     (TypeGrasp ?grasp)
           )
           :outputs (?traj ?q)
           :certified (and (IsCFreePathFromHome ?traj ?q)
                           (KinGrasp ?object ?pose ?grasp ?q)
                           (TypeConf ?q))
  )
  (:stream sample-path-relocate
           :inputs (?object ?pose1 ?grasp ?q1 ?pose2 ?q2)
           :domain (and
                     (IsGraspable ?object)
                     (TypePose ?pose1)
                     (TypePose ?pose2)
                     (TypeGrasp ?grasp)
                     (TypeConf ?q1)
                     (TypeConf ?q2)
                     (IsGrasp ?grasp ?object)
                     (KinGrasp ?object ?pose1 ?grasp ?q1)
                     (KinGrasp ?object ?pose2 ?grasp ?q2)
           )
           :outputs (?traj)
           :certified (IsCFreePathGrasping ?object ?grasp ?q1 ?traj ?q2)
  )
)

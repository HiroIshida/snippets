(define (stream tamp) 
  (:stream sample-grasp-pose
           :inputs (?object)
           :domain (IsGraspable ?object)
           :outputs (?pose)
           :certified (and (IsGrasp ?pose ?object) (IsPose ?pose))
  )
  (:stream sample-path-to-pose ;; solve IK to pose and motion plan
           :inputs (?q1 ?pose)
           :domain (and (IsConf ?q1) (IsPose ?pose))
           :outputs (?traj ?q2)
           :certified (and (IsCFreePath ?q1 ?traj ?q2) (kin ?q2 ?pose))
  )
)

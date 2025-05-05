(define (domain reach)
  (:requirements :strips :equality)
  (:predicates
    (kin ?q ?pose)
    (AtConf ?q)
    (IsConf ?q)
    (IsPose ?pose)
    (IsCFreePath ?q1 ?traj ?q2)
    (IsGraspable ?pose)
    (IsGrasp ?pose ?object)
  )
  (:action hand_free_move
    :parameters (?q1 ?traj ?q2)
    :precondition (and
        (AtConf ?q1)
        (IsCFreePath ?q1 ?traj ?q2)
    )
    :effect (and
        (not (AtConf ?q1))
        (AtConf ?q2)
    )
  )
)

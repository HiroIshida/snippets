- solve pomdp. 
- goal set is defined by a set of belief that only contains one element. 
- 経験したbelief state set B^E とgoal setのunionをとり, その中でのjampをpenalizeする.
- online exploitation: b_startから experience \pi にしたがって, 遷移し, 通過したB^E の集合を記録する.

## question
- what is the form of the \pi? Is it a sequence of a belief state or a tree of an action of sequence.
- memory consumption and computational cost of pre-processing.
- Hのサイズはどれくらい? even if you take advantage of the experience, the size of H is still large so that it is difficult to solve the POMDP.
- H is a power set of all the possible pose set?? Or you confine the size of H by some criteria?
- In the library construction, you retrieve the similar experience from the library based on the similarity of the uncertainty. How do you define the similarity of the uncertainty?

# pr2のデバッグ
rosrun rqt_pr2_dashboard rqt_pr2_dashboard

# noeticのはまりどころ
- python3_is_pythonをいれないといけない. #!/bin/env pythonでpythonがないとおこられる. 
- rqt_pr2_dashbourdは最新のものを入れないといけない. -> pr2_self_testからpr2_motor_diagnostic_toolをソースビルドする必要あり

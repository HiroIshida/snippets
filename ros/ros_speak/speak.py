# -*- coding: utf-8 -*-

import actionlib
from collections import defaultdict
import os.path as osp
import math
import rospy
import pandas as pd

from sound_play.msg import SoundRequestAction
from sound_play.msg import SoundRequest, SoundRequestGoal
from power_msgs.msg import BatteryState

from diagnostic_msgs.msg import DiagnosticArray

rospy.init_node("battery_warning")
speak_client = actionlib.SimpleActionClient("/robotsound_jp", SoundRequestAction)
waitEnough = speak_client.wait_for_server(rospy.Duration(10))

req = SoundRequest()
req.command = SoundRequest.PLAY_ONCE
req.sound = SoundRequest.SAY
req.arg = "おーい。きたがわ"
req.arg2 = "ja"
req.volume = 1.0
speak_client.send_goal(SoundRequestGoal(sound_request=req))
speak_client.wait_for_result(timeout=rospy.Duration(10))

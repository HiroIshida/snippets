from mujoco_xml_editor import MujocoXmlEditor
import mujoco
import mujoco_viewer

editor = MujocoXmlEditor.load("./cabinet.xml")
editor.add_sky()
editor.add_ground()
editor.add_light()
xmlstr = editor.to_string()

# write to file
with open("./hoge.xml", "w") as f:
    f.write(xmlstr)

model = mujoco.MjModel.from_xml_string(xmlstr)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)
while True:
    mujoco.mj_step(model, data)
    viewer.render()

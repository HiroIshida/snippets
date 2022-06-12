import trimesh

file_path = "/home/h-ishida/.randblend/gso_dataset/ASICS_GELAce_Pro_Pearl_WhitePink/visual_geometry.obj"
mesh = trimesh.load_mesh(file_path)
scene = mesh.scene()
png = scene.save_image(resolution=[400, 200], visible=True)
with open("tmp.png", 'wb') as f:
    f.write(png)
    f.close()

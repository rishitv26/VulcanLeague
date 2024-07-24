from config import Config

obj = Config()

obj.read_configs()
obj.edit("test","ragav is an indiot")
print(obj.get("test"))
obj.add("if ragav = exist, then ragav","ARCHAIC NONSENSE FROM THE 1ST CENUTRY B.C.")
obj.save()

print(obj.SETTINGS)

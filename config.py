
class Config:
    def __init__(self):
        self.file_name = "config.txt"
        self.SETTINGS = {}
        self.read_configs()
    
    # reading from config.txt to get settings.    
    def read_configs(self):
        try:
            file = open(self.file_name, "r")
        except:
            file = open(self.file_name, 'w')
            file.close()
            file = open(self.file_name, "r")
        
        for line in file.readlines():
            try:
                data = line.split("=")
            except:
                continue
            for i in range(len(data)):
                data[i].replace('\n', '')
            self.SETTINGS[data[0]] = data[1].rstrip('\n')
        
        file.close()
    
    def edit(self, setting: str, new_value: str):
        if not setting in self.SETTINGS:
            raise ValueError("ERROR, setting '" + setting + "' does not exist.")
        else:
            self.SETTINGS[setting] = new_value
    
    def get(self, setting: str):
        if not setting in self.SETTINGS:
            raise ValueError("ERROR, setting '" + setting + "' does not exist.")
        else:
            return self.SETTINGS[setting]
        
    def add(self, setting: str, new_value: str):
        if setting in self.SETTINGS:
            raise ValueError("ERROR, setting '" + setting + "' already exists.")
        else:
            self.SETTINGS[setting] = new_value
    
    def save(self):
        file = open(self.file_name, 'w')
        file.write("")
        file.close()
        file = open(self.file_name, "a")
        for key in self.SETTINGS:
            file.write(key + "=" + self.SETTINGS[key] + "\n")
        file.close()



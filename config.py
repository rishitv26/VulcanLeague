
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
        
        for line in file.readlines():
            try:
                data = line.split("=")
            except:
                continue
            for i in range(len(data)):
                data[i].replace('\n', '')
            self.SETTINGS[data[0]] = data[1].rstrip('\n')
        
        file.close()
    
    def edit(self, )
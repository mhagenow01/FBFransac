import os
import pickle

class OnDisk:
    def __init__(self, folder):
        self.Folder = folder
        
        if not os.path.isdir(self.Folder):
            os.mkdir(self.Folder)

    def isCached(self, key):
        return os.path.isfile(os.path.join(self.Folder, key))
    
    def getCachedVal(self, key):
        with open(os.path.join(self.Folder, key),'br') as fin:
            return pickle.load(fin)
    def cacheVal(self, key, val):
        with open(os.path.join(self.Folder, key),'bw') as fout:
            pickle.dump(val, fout)

    def cache(self, func, key):
        def cachedFunc(*args, **kwargs):
            if self.isCached(key):
                return self.getCachedVal(key)
            val = func(*args, **kwargs)
            self.cacheVal(key, val)
            return val
        return cachedFunc



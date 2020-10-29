import time

class DebugUtils():

    instance = None

    @classmethod
    def get_instance(cls, millis_show_FPS=0, logfilename=None):
        if DebugUtils.instance is None:
            DebugUtils.instance = DebugUtils(millis_show_FPS, logfilename)
        return DebugUtils.instance


    def __init__(self, millis_show_FPS, logfilename=None):
        self.millisShowFPS = millis_show_FPS
        self.logfilename = logfilename
        self.open_file()
        self.millisInitFPS = 0
        self.fps = 0
        self.millisEndFPS = 0

    def set_logfilename(self, filename):
        self.logfilename = filename

    def open_file(self):
        if self.logfilename is not None:
            self.logfile = open(self.logfilename, 'w')

    def msg(self, *args):
        str_msg = ''
        for each in args:
            str_msg += str(each)
        print(str_msg)
        if self.logfilename is not None:
            self.logfile.write('\n'+str_msg)

    def initFPS(self):
        self.millisInitFPS = time.time()


    def showFFS(self, discountTime=0):
        millisThis = time.time() - self.millisInitFPS
        if self.millisEndFPS >= self.millisShowFPS:
            self.fps = self.fps / self.millisEndFPS
            self.msg('FPS: ', self.fps)
            self.millisEndFPS = 0
            self.fps = 0
        else:
            self.millisEndFPS += millisThis
            self.millisEndFPS -= discountTime
            self.fps += 1


    def flush_file(self):
        if self.logfilename is not None:
            self.logfile.flush()


    def close_file(self):
        if self.logfilename is not None:
            self.logfile.close()
        DebugUtils.instance = None


if __name__ == '__main__':
    debug = DebugUtils.get_instance(0, "/dados/tmp.txt")
    debug.msg("Hello;")
    debug.flush_file()
    debug.close_file()
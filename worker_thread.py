from environment import Environment

class Worker_thread:
    # スレッドは学習環境environmentを持ちます
    def __init__(self, env, args):
        self.environment = Environment(env, args)
        #self.thread_type = thread_type

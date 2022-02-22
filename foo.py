class Nop(object):
    def log(*args, **kw):
        print('args', args)
        print('kwargs', kw)

    def init(*args, **kw):
        print('args', args)
        print('kwargs', kw)

    def Image(x):
        return x
        
    def __getattr__(self, _): 
        return self.nop
def deco(fun):
    def decofun(**kwargs):
#        print(kwargs.get('c'))
        print(kwargs.get('c'))
        return fun(kwargs.get('x'),kwargs.get('y'))
    return decofun


class MM():    
    @deco
    def rr(x,y):
        print('rrr')
        return x+y
print(MM.rr(c=5,x=4,y=3))
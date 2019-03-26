#coding=utf-8

class A(object):
    def add(self,x):
        y = x+1
        print y

class B(A):
    def add(self,x):
        super(B,self).add(x)

# b = B()
# b.add(2)

#########类的区别##########
#旧式类
class People:
    pass

#新式类
class Person(object):
    pass

a = People()
print dir(a)
b = Person()
print dir(b)

#注：旧式类和新式类主要区别于python2.X，在python3.X中默认为新式类（不管后面带不带object）
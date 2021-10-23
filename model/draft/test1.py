
class Base():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def p(self):
        print(self.mo)


class Son(Base):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.mo = 'model'


s = Son(2, 3)
s.p()

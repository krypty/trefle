class CocoCouple:
    def __init__(self, sp1, sp2, fitness):
        self.sp1 = sp1
        self.sp2 = sp2
        self.fitness = fitness

    def __eq__(self, other):
        if self.sp1.bits != other.sp1.bits:
            return False
        elif self.sp2.bits != other.sp2.bits:
            return False
        else:
            return True

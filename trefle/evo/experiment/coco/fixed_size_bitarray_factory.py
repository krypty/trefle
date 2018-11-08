from random import randint

from bitarray import bitarray


class FixedSizeBitArrayFactory:
    @staticmethod
    def create(n_bits):
        class FixedSizeBitArray:
            def __init__(self, bin_str=None):
                if bin_str is None:
                    bin_str = format(
                        randint(0, (2 ** n_bits) - 1), "0{}b".format(n_bits)
                    )
                self.bits = bitarray(bin_str)

            def deep_copy(self):
                other = self.__class__(self.bits.to01())
                return other

            def __len__(self):
                return self.bits.length()

            def __setitem__(self, key, value):
                self.bits.__setitem__(key, value.bits)

            def __getitem__(self, item):
                instance = FixedSizeBitArray()
                instance.bits = self.bits[item]
                return instance

        return FixedSizeBitArray

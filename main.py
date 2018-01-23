# def main():
#     print("Please see examples before starting in the examples folder")
from evo.playground.ifs_without_evo import main


def main2():
    from time import time

    t0 = time()
    for i in range(10):
        main()

    print("yolo", (time() - t0) * 1000)


if __name__ == '__main__':
    main2()

from matplotlib import pyplot as plt

from pyfuge.fuzzy_systems.examples.car_problem_slides.car_problem_default_rule import \
    car_accel_problem as cap_default_rule
from pyfuge.fuzzy_systems.examples.car_problem_slides.car_problem_default_rule_and_not import \
    car_accel_problem as cap_default_rule_and_not
from pyfuge.fuzzy_systems.examples.car_problem_slides.car_problem_mamdani import \
    car_accel_problem as cap_mamdani
from pyfuge.fuzzy_systems.examples.car_problem_slides.car_problem_singleton import \
    car_accel_problem as cap_singleton
from pyfuge.fuzzy_systems.view.fis_surface import show_surface


def main():
    fis_mamdani = cap_mamdani()
    fis_singleton = cap_singleton()
    fis_default_rule = cap_default_rule()
    fis_default_rule_and_not = cap_default_rule_and_not()

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    show_surface(fis_mamdani, title="Mamdani",
                 x_label="speed", y_label="speed_change",
                 z_label="action",
                 n_pts=15, x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1),
                 ax=ax)

    show_surface(fis_singleton, title="Singleton",
                 x_label="speed", y_label="speed_change",
                 z_label="action",
                 n_pts=15, x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1),
                 ax=ax2)

    show_surface(fis_default_rule, title="Singleton + DR",
                 x_label="speed", y_label="speed_change",
                 z_label="action",
                 n_pts=15, x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1),
                 ax=ax3)

    show_surface(fis_default_rule_and_not, title="Singleton + DR + not",
                 x_label="speed", y_label="speed_change",
                 z_label="action",
                 n_pts=15, x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1),
                 ax=ax4)

    plt.show()


if __name__ == '__main__':
    main()

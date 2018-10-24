class CocoEvolutionViewer:
    @staticmethod
    def plot_fitness(logbook):
        import matplotlib.pyplot as plt

        gen = logbook.select("gen")
        gen = list(set(gen))
        fit = logbook.select("species", "max")
        max_fit_sp1 = [v[1] for v in zip(*fit) if v[0] == "sp1"]
        max_fit_sp2 = [v[1] for v in zip(*fit) if v[0] == "sp2"]

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, max_fit_sp1, "b-", label="Max fit sp1")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()

        line2 = ax2.plot(gen, max_fit_sp2, "r-", label="Max fit sp2")
        ax2.set_ylabel("Fitness sp2", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.show()

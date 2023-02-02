import ga

if __name__ == "__main__":
    pop = ga.SNN_GA(1000, 32, 10000)
    print(pop.mask[0, 0, :20])
    pop.run()
    print(pop.mask[0, 0, :20])
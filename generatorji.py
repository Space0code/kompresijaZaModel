def generiraj_soda_stevila_do_n(n):
    for i in range(1, n + 1, 2):
        yield i + 1


for x in generiraj_soda_stevila_do_n(10):
    print(x)

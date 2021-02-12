import torchvision as tv


def main():
    df = tv.datasets.CIFAR10("./datasets/", download=True)
    print(type(df.data[0]))
    for i in range(0, 27, 0):
        print(i)
    # r = LinearRegression()


if __name__ == "__main__":
    main()

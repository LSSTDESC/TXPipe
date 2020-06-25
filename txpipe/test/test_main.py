import ceci.main


def test_example():
    ceci.main.run("./examples/laptop.yml", ["resume=False"])

if __name__ == '__main__':
    test_example()

import ceci.main


def test_example():
    status = ceci.main.run("./examples/laptop_pipeline.yml", ["resume=False"])
    assert status == 0

if __name__ == '__main__':
    test_example()

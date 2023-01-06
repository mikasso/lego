import random
import sys

from lego_record import LegoRecord

labels = [f"lego{i}" for i in range(432)]


def generate_path() -> str:
    index = random.randint(1000, 9999)
    return f"./image_{index}"


def generate_record() -> LegoRecord:
    path = generate_path()
    label = random.choice(labels)
    features = [str(random.randint(0, 1000)) for _ in range(10)]
    return LegoRecord(path, label, features)


def generate_file(num_of_rows: int, filename: str) -> None:
    records = [generate_record() for _ in range(num_of_rows)]
    with open(filename, "w") as f:
        for record in records:
            f.write(str(record) + "\n")


if __name__ == "__main__":
    num_of_rows = 100 if len(sys.argv) <= 1 else int(sys.argv[1])
    filename = "./data.csv" if len(sys.argv) <= 2 else sys.argv[2]
    generate_file(num_of_rows, filename)

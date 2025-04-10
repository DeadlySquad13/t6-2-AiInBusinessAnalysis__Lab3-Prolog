<!-- toc -->

# Big Data
## 1. MapReduce Concept using built-in methods.
### Generating Big Data
Run `generate-data` script, for example:
```py
# Generate 1 million rows in ./data/test.csv
pixi run generate-data ./data/test.csv --number_of_rows 10000000

# Generate array of test files in ./data directory with rows specified in
# array `rows`:
rows=( 100 1000 10000 100000 1000000 2000000 3000000 4000000 5000000 )
for i in "${rows[@]}"; do pixi run generate-data ./data/"test-$i.csv" --number_of_rows "$i"; done
```

To see full list of options use help:
```py
pixi run generate-data --help
```

### Get Statistics
```bash
pixi run thread-pool-executer ./data/test-1000000.csv
# Or
pixi run thread-pool-executer ./data/test-1000000.csv --top_n 10 --chunsize 10000
```

To see full list of options use help:
```py
pixi run thread-pool-executer --help
```

## 2. Apache Hadoop
Run:
```bash
make prepare-hadoop
make build-hadoop
make run-hadoop
```

Open admin panel in browser using `make open-hadoop-admin-panel`

> If you have problems opening site check if you have any Nginx settings
messing up you connection.

## Contributing
Created using Base with Docker Project bootstrap template
Has [Base template](https://github.com/dsOmega-bootstrap/Base.bootstrap) features and following:
- Simple Docker Makefile with QoL targets
- Docker .env for easier configurability of commands in aforementioned Makefile

### How to Use
See [.env.example](./.env.example). Copy it to `.env` and fill.

## Contributing
See guidelines in [Contributing](./CONTRIBUTING.md). This projects also
has Docker support, see ["Running in Docker"
section in Contributing](./CONTRIBUTING.md#running-in-docker).

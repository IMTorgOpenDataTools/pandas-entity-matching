# Entity Matcher for Pandas


## Usage

Main entrypoint to the program:

```
pipenv install
pipenv run python cbdc_scraper/scraper.py
```


## Testing

Basic testing can be performed using `pytest`.

Command line testing of a specific test method can be performed using: `pytest --trace tests/test_all.py -k test_all`

Use the following commands:

    n(next) – step to the next line within the same function
    s(step) – step to the next line in this function or called function
    b(break) – set up new breakpoints without changing the code
    p(print) – evaluate and print the value of an expression
    c(continue) – continue execution and only stop when a breakpoint is encountered
    unt(until) – continue execution until the line with a number greater than the current one is reached
    q(quit) – quit the debugger/execution




## References

* Adapted from basic workflow provided by Tomonori Masui using [notebook](https://github.com/tomonori-masui/entity-resolution/blob/main/entity_resolution_implementations.ipynb)
* Original dataset from University of Leipzig, Database Group Leipzig, [Benchmark datasets for entity resolution](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution)
# Parser Registry

The `battery_analysis.parsers` package uses a simple registry to map file
extensions to parsing functions.  Built-in parsers register themselves when
the package is imported, and `parse_file()` uses the registry to dispatch to
the correct handler.

## Registering a custom parser

Third-party code can add support for new file types without modifying the
core package by registering a parser function:

```python
from battery_analysis.parsers import register_parser

def parse_custom(path: str):
    ...  # parse the file and return (cycles_summary, metadata)

register_parser(".myext", parse_custom)
```

After registration, `parse_file()` will automatically use `parse_custom` for
files ending with `.myext`.


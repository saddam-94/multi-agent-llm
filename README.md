Multi Agent llm for  identifies and analyzes competitors for given input startup or product query.

## Installation

create a conda or python env.

```bash
conda create -n multi-agent python=3.10 anaconda
```

run the below cmd for install the dependencies
```bash
pip install -r requirement.txt
```

## Usage

invoke the main method from Main class.
```python
from main import Main

main = Main()

main.main("AI Assistant")

```

If you want to reproduce the source please follow the below cmd

```
python main.py --device 'cuda' --query "pizza"
```

if you want run the default device and query just run below cmd

```bash
python main.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

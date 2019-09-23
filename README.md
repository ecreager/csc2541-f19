# csc2541-fall-2019
[CSC 2541F: AI and Ethics: Mathematical Foundations and Algorithms Fall,
2019](http://www.cs.toronto.edu/~toni/Courses/Fairness/fair.html)

Virtual environment instructions (optional, requires Python 3.6+):
```
#!/bin/bash
export VENVDIR=~/venv/csc2541-f19
mkdir $(dirname $VENVDIR)
mkdir $VENVDIR
python3 -m venv $VENVDIR
source $VENVDIR/bin/activate  # use this command to "activate" the environment
pip install --upgrade pip
pip install -r requirements.txt
echo done with setup
```

Executing the above commands may take some time.
Halting `pip` commands using `Ctrl+C` is _not_ recommended.

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

## Note on forking this repo
Note that this is a public repo, which means that _all_ folks of this repo are also public. 
Therefore any solution code you write _should not_ be pushed to a public fork (otherwise other students can see your solution!). 
* If you _do_ want to version your code and _do_ want to host it privately on github (this allows you to view source code on github.com, sync across machines, etc.), then you should follow the instructions below to duplicate (not fork) the repo.
This will create a private copy where you can push freely.
https://help.github.com/en/articles/duplicating-a-repository
* If you _do_ want to verison your code but _do not_ care about hosting it on github, then you can simply clone this repo (instead of forking it). You can still use git locally (commit, branch, etc.), but you cannot push local changes since you do not have contributor permissions to this repo.
* If you _do not_ care about versioning your code then you can simply download this repo as a .zip folder and not worry about git.

# Steps to sync to gitHub
git status
git add .
git commit -m "Msg here"
git push origin master
---
# Steps to create "snapshot in time" for each episode
## freeze progress for an episode, create a branch:
git checkout -b episode-1-yadayada
git push origin episode-1

## Switch back to the main branch to continue working
git checkout main

---
# Steps to view old v   ersion of file

git log --follow -- src/Metrics.py

## show commits where file has changed with nice formmating of date
git log --follow --pretty=format:"%h %ad | %s" --date=short src/Metrics.py

## View differences
git diff <commit_hash> <path_to_file>
git diff f8db2ee798a22cad26fd872170e976276a37bd15 /src/Metrics.py

## restore old version to different file.
git show <commit_hash>:<path_to_file> > <Path to file of different name>
git show f8db2ee798a22cad26fd872170e976276a37bd15:src/Metrics.py > src/Metrics_old_version.py


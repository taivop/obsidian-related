# Obsidian related notes

This repository contains a Python app that serves related notes to the [obsidian-lab](https://github.com/cristianvasquez/obsidian-lab) plugin and shows up in the sidebar.

The related note search is very heuristic and currently relies only on the graph structure and simple features of the file (title length, file length, etc). It does not attempt to do NLP on the text; this is the next step of this project.

### Installation

This package requires
* Python 3.9 (3.7+ should work but has not been tested)
* `pip install -r requirements.txt`

### Usage
Clone this repository, and then in the repo root run the prediction server:
```
$ uvicorn api:app --port 5000 --host "localhost" --reload
```

The server needs to be running for predictions to be served to the Obsidian app.

After the server is running, install and enable the [obsidian-lab](https://github.com/cristianvasquez/obsidian-lab) plugin and you should see an action called `related`. Calling this action (which you can also do from the command palette), in the sidebar under the tab with the icon you specified in obsidian-lab settings.

I recommend enabling the trigger "triggers when opening a file" so the predictions are automatically made every time you open a note, so you don't have to manually call the function.

### Performance
Predictions for a file take 50-100 milliseconds per query on my vault with ~500 notes, on a 2015 i7 Macbook Pro.

### Contribution
This project is in a very early stage and I might make massively breaking changes at any time. Let's discuss in Github issues if you have feature requests or would like to contribute.

### TODOs

* [x] Ability to force-reindex when predicting
* [x] Remove fixed/magic paths and numbers from the codebase and put them into env
* [x] Profile and make faster (<0.5 seconds ideally)
* [x] NLP-based predictions -- first attempt based on [top2vec](https://github.com/ddangelov/Top2Vec)
* [ ] gitignore-like ignoring of specific files/patterns
* [ ] Make obsidiantools ignore case
* [ ] Handle if linked note is renamed => vault index needs update

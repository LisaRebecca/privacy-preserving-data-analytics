# privacy-preserving-data-analytics
 
## Commands
- Export your conda env: `conda list -e > requirements.txt`

- Load conda env: `conda create -n torch-env --file requirements.txt`

- Run training: `cd src && python main.py``

- Run training with params: `python main.py --subset_size=1000 --model=ResNet --epochs=10 --optimizer=SGD`


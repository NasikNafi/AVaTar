# AVaTar
Code for the paper "Policy Optimization with Augmented Value Targets for Generalization in Reinforcement Learning" published at IJCNN 2023.

# Dependencies
Run the following to create the environment and install the required dependencies: 
```
conda create -n AVaTar python=3.7
conda activate AVaTar

cd AVaTarg
pip install -r requirements.txt

pip install procgen

pip install protobuf==3.20.0

git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 
```


# Instructions 

### To Train AVaTar on Bigfish
```
python train.py --env_name bigfish
```


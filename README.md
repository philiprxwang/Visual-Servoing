# Visual-Servoing using PyBullet 

Create environment and install packages

```
conda create --name panda python==3.10
conda activate panda
pip install -r requirements.txt
```

Test

```
python pybullet_panda_camera.py
```

To test gym custom environment

```
python test_script.py
```

To test DQN from SB3 on custom environment

```
python train_and_test_dqn_sb3.py --stacking True
```
```
python train_and_test_dqn_sb3.py --stacking False
```

To test DQN implementation 
```
python train_dqn.py
```

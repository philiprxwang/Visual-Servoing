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

To test A2C SB3 on custom environment

```
python train_and_test_a2c_sb3.py
```

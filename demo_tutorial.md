# Demo Tutorial

This tutorial is for the [final_pred.py](final_pred.py) program, which is the demo program for this project.

Because the "match" statement in "jax" is syntax introduced in Python 3.10 and later, please make sure this project is configured to use [Python 3.10](https://www.python.org/downloads/release/python-3100) and execute the following commands to demonstrate.

---

Create a virtual environment:

```bash
py -3.10 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/Scripts/activate
```

Upgrade pip (optional):

```bash
python -m pip install --upgrade pip
```

Install dependencies (may be slow in Asia):

```bash
pip install -r requirements.txt
```

You can use a specific mirror to speed up installation:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

Run the demo program:

```bash
python final_pred.py
```

Sign language:

![Sign-Language](https://user-images.githubusercontent.com/99630855/201489493-585ffe5c-f460-402a-b558-0d03370b4f92.jpg)

Note: You need to perform the "next" gesture (open five fingers) to add the Current Symbol to the Sentence.

---

Normally, the demo should run after installing as described above.

If dependencies need to be reinstalled due to version issues (e.g. reinstall opencv):

```bash
pip uninstall opencv-python -y
```

You can clear the cache before reinstalling dependencies:

```bash
pip cache purge
```

When reinstalling dependencies it's recommended to use `-i https://pypi.tuna.tsinghua.edu.cn/simple` to speed up:

```bash
pip install opencv-python==4.5.5.64 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

To check the version:

```bash
pip show opencv-python
```

# Project: Recommendation System for Movie Ratings

Generates a graph visualization of feature importance.

[Project reference](https://medium.com/towards-artificial-intelligence/recommendation-system-in-depth-tutorial-with-python-for-netflix-using-collaborative-filtering-533ff8a0e444)

### Requirements

```
certifi==2020.6.20
cycler==0.10.0
joblib==0.17.0
kiwisolver==1.2.0
matplotlib==3.3.1
numpy==1.19.1
pandas==1.1.1
Pillow==7.2.0
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.1
scipy==1.5.4
seaborn==0.11.0
six==1.15.0
sklearn
threadpoolctl==2.1.0
xgboost===1.3.0-SNAPSHOT
```

### Installation

Project requires: 

   - [CMake](https://cmake.org/)
   - [Visual Studio](https://visualstudio.microsoft.com/downloads/)
   - [virtualenv](https://pypi.org/project/virtualenv/#description)
   - [xgboost](https://xgboost.readthedocs.io/en/latest/build.html)

Run install.bat from the root of the project.

```cmd
echo cloning

git clone --recursive https://github.com/dmlc/xgboost

echo Building xgboost
cd xgboost
mkdir build
cd build
cmake .. -G"Visual Studio 14 2015 Win64"
REM # for VS15: cmake .. -G"Visual Studio 15 2017" -A x64
REM # for VS16: cmake .. -G"Visual Studio 16 2019" -A x64
cmake --build . --config Release

echo installing requirements

cd ../../
pip install -r requirements.txt

echo installing xgboost

cd xgboost/python-package

python setup.py install

echo complete
```

# License

MIT

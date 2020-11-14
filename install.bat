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
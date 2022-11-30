rm -rf food-101
wget --no-check-certificate http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvf food-101.tar.gz
rm -rf food-101.tar.gz
python prepare_data.py
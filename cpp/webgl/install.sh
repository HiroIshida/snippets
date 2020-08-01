BASEDIR=$(pwd)
cd /tmp
git clone https://github.com/google-research/tiny-differentiable-simulator.git
cp -r tiny-differentiable-simulator/third_party $BASEDIR
rm -rf tiny-differentiable-simulator


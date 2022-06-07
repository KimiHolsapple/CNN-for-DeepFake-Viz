# CNN-for-DeepFake-Viz
# fake_img_det.py

fake_img_det.py is a Python script meant to use convolution neural networks in order to create amazing visualizations to gain deeper insights into DeepFakes. It utilizes Meso-4, a CNN model presented by researchers in 2018 and extends it create feature maps and Grad-CAM vis. Output is saved to a pdf file in the same directory as the script called "viz_output.pdf". 

## Requirements

Ensure you the file Meso4_DF in the same directory as the script and you are good to go.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary libraries.

```bash
pip install numpy
pip install matplotlib
pip install tensorflow
pip install keras-vis
pip install scipy.ndimage
pip install PIL
pip install sys
```
Make sure Meso4_DF is in the same directory and so are the DeepFake and Real data set files. For you this will look like  

## Usage

```python
python3 fake_img_det.py

```
When prompted follow the menu instructions and select 1 or 2 to either read files from a directory, or to read a single file path. If reading from a directory it will look like ~/YOU_USER/final_project.

If selecting option 2 you can load any image you want and make sure you know the classifier if you want to know if the model predicted it correctly. Classifier is "0" for DeepFake and "1" for Real.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Support

E-Mail Kimi Holsapple at kholsapp@ucsc.edu

## License
[MIT](https://choosealicense.com/licenses/mit/)

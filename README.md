# AMBR
Automated Mouse Behavior Recognition

This repository includes all the python utilities for the project described in my article for the refereed [VAIB 2016 workshop](http://homepages.inf.ed.ac.uk/rbf/vaib16.html), held in conjunction with the ICPR 2016 conference:

Kramida, Gregory, Yiannis Aloimonos, Chethan Mysore Parameshwara, Cornelia Fermüller, Nikolas Alejandro Francis, and Patrick Kanold. "Automated Mouse Behavior Recognition using VGG Features and LSTM Networks." [Link to PDF](http://homepages.inf.ed.ac.uk/rbf/VAIB16PAPERS/vaibkramida.pdf).

### Potentially Useful Code

There are two major code contributions you can utilize here:

* Class structure / system for custom video processing. The IMBGS background subtractor[1] which is used within these scripts is [cve, a separate opencv-based C++ utility](https://github.com/Algomorph/cve) that is ported to python. You can find it here, compile using CMake, and run make_install. Credit goes to [Domenico Daniele Bloisi](http://www.dis.uniroma1.it/~bloisi/software/imbs.html) for the original IMBS implementation that I adapted for this project. It also uses the vgg_utils.py script for extracting VGG features [2] from the foreground bounding boxes of each video frame. It was adapted from [this repository](https://github.com/jesu9/VGGFeatExtract) by @jesu9.

* Complete LSTM network setup, with an embedding/classifier layer. This includes many useful tools for improving performance in various situations, such as "Dropout" and ADADELTA (see workshop article for details and references.) The input features that it expects is a numpy archive with a matrix called "features", where each row is a feature vector representing a single time step. The label file that it expects (for basic, single-view mode) is a simple json file. See format from example given below after the Dependencies section. For a dataset, both files should have the same name but different extension (i.e. .npz and .py). Multiple datasets may be used at once to train the network. The LSTM network code easily generalizes to any scenario with discretized time/event sequences without any modifications. The original bare-bone LSTM implementation that this code is based on is [one by Pierre Luc Carrier or Kyunghyun Cho](http://deeplearning.net/tutorial/lstm.html).

Note that the program is currently tested well only for single-view scenarios, i.e. where data is not multimodal or coming from multiple views on the same subject. The multiview mode in the LSTM code is purely experimental at this point, as I currently lack sufficient data to test it. Heck, I bet that it's buggy, use multiview stuff at your own peril.

### Dependencies

This cose was only tested on Ubuntu, albeit it is writ in Python **3** (and CMake-based C++ for the cve library if you need that part), so theoretically it should be usable on other major operating systems. Most definitely you can use it on any debianoid system.

The prerequesites are as follows:
OpenCV, Caffe, and cve for various parts of (1) above. Caffe & OpenCV should be compiled correctly using Python 3. This may require you to ensure you have the latest protobuf (3) and it's corresponding Python 3 bindings (can be installed via pip), both of which should be easily installable or included in your system as long as you keep up with the times.
For LSTM (part 2 above), you only need an up-to-date numpy, matplotlib, and theano, all of which are easily installable via pip.

### JSON label file format
```json

[
   {
      "start": 0,
      "end": 2098,
      "label": 0
   },
   {"...More entries go here..."},
   {
      "start": 34244,
      "end": 34598,
      "label": 0
   },
   {
      "start": 34599,
      "end": 34678,
      "label": 2
   }
]
```
Each entry within the array represents a sequence, the [start, end] bounds are just 0-based indexes (bounds are both inclusive!). These point to the rows in the feature matrix that should be provided in the numpy archive.

To ease manual annotation, the label_converter.py script is also provided, which can convert labels from a manual text-entry format to the above JSON format. Here is an example of a manual text-entry label file:
```
3131-3162:G
14729-15313:[out of frame]
16130-16151:G +0
16336-16347:G +2
16665-17446:[out of frame]
19224-19284:R?
19453-19477:G

//more entries here

28788-29000(?):R
29001-29578:[out of frame]
29698-29790:R
30092-30252:R

//more entries here

31451-31470:R
31795-31943:R
32064-32088:G
32249-32383:R
32750-33399:[out of frame]
34231-34243:G +16
34599-34678:R
```
Here, each entry denotes a sequence interval, and each letter that follows it denotes a label. The label number and designations can be amended or altered by changing the label_mapping and inverse_label_mapping dictionaries within label_converter.py. Pay attention to the default_label flag and the default_behavior_label, which is set in main! This will constitute which label gets assigned to the intervals *between* the ones that were manually labeled.

The "+2" denotes an increment in frames (may be negative as well). If the video frame rate is unstable, as it is with many cheap consumer-end cameras, this allows to manually adjust the time. This is only relevant for "multiview" mode of operation, where features are concurrently gathered from multiple cameras around the subject, and will not reflect on "single-view" label file generation.

### Runnables

Useful runnable utilities are:
* extract_vgg_features_from_video.py (compiles feature numpy archives)
* extract_silhouettes.py (compiles BG-subtracted video of single moving subject & corresponding mask video -- useful for debugging background subtraction)
* label_converter.py (for converting labels from simpler text files to properly-formatted json, see section above)
* lstm/run.py (The actual LSTM training/testing procedure.)
  
[1] Bloisi, Domenico, and Luca Iocchi. "Independent multimodal background subtraction." In CompIMAGE, pp. 39-44. 2012.
[2] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

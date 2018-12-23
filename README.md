# SEN Virtual Mouse Program

Virtual mouse program with OpenCV written in Python. Goal is to create a program that allows control of the mouse with natural hand gestures that mimic those used to control a actual computer mouse. Uses Haar Cascade to detect the hand and track the hand, creating a mask that displays only the hand. 2 pieces of colored tape (color 1) on the palm of the hand that allow for mouse movement as well as scaling, and 1 piece of colored tape (color 2) on the first finger to allow for left mouse button control.

You can see a screenshot of what the final result looks like below.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need Numpy (general all round use), Pynput (for mouse control), OpenCV-Python (for computer vision), Matplotlib (displaying histograms), imutils (frame resizing) and Scipy (vector distance calculations).

To get these you can do the following in you command line (I have windows so I am using cmd in this case)

```
pip install numpy
pip install pynput
pip install opencv-python
pip install matplotlib
pip install imutils
pip install scipy
```

### Set-Up

Before running the code you will have to run through it and adjust the parameters depending on your environment, preferences and available colors. I personally use a green and blue electric tape. To run the program just run the code through your IDE (I use pycharm) or initiate it through the command line.

The process of setting up the colors involves taking screenshots of the colored tape attached to your hand in the scene in which you will be using the mouse, compiling these screenshots into a single image and placing that compiled image in the relevant path (hand/color1/COLOR1_final.jpg) or (hand/color2/COLOR2_final.jpg) and finally touching up be directly bounding the colours on a displayed histogram. Ill run you through an example:

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

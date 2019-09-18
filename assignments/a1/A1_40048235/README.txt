The program is written by using CLion on Mac.
There is only one .cpp file, which is main.cpp, and it takes no program arguments.

The image (relative) addresses are hard coded on lines 120 ~ 122, change variable testImageIndex from 0 ~ 2 on line 123 to change the reading images.

All combined images are stored in "results" folder.
The close-up compared images are marked, and also stored in "results" folder.


Note:
1) The CMakeLists.txt file may be change if the program is run on another machine.
2) The submitted version has already been builded. To run it, go to cmake-build-debug fodder, issue './cvA1' in terminal. In order to read another file, change the main.cpp on line 123, then re-build.
3) To re-build, in terminal, create another folder, go in there, copy image_set folder in, issue 'cmake ../', then issue 'make', run with './cvA1'. I don't know why it cannot be rebuild in cmake-build-debug folder, maybe it is generate by the CLion, it has some unique file path or what.
4) Since the imwrite() functions are not removed, so for each run, it should generate some new images in local folder, it can be ignored.
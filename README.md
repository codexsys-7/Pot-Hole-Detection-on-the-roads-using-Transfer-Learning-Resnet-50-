![pothole](https://media1.tenor.com/images/824941b982135601c3f496e1caeec2ba/tenor.gif?itemid=14445555)

# _**Pot Hole Detection on the roads using Transfer Learning Resnet50**_
As we all know how pot holes play a role in causing most of the minor and some major accidents actors the globe, having potholes on the roads are very dangerous in any climate, since it can cause a very serious damage to vehicle’s and even the drivers which can make the vehicle fall apart and causing serious injuries. So, here is a project which allows us to solve the problem of avoiding potholes where the model which we have generated using transfer learning i.e., Resnet50 is used to detect whether there are any pot holes or not on the roads, this model can be integrated with autopilot vehicle modes which allows the vehicle to avoid potholes.

# _**Base Paper**_
+ https://www.researchgate.net/publication/357093620_Smart_Pothole_Detection_Using_Deep_Learning_Based_on_Dilated_Convolution

# _**Algorithm Description**_
Resnet is a Transfer learning algorithm which is mainly known as Residual Neural Networks which helps us to solve the problem of vanishing gradients/exploding gradient descents which arises while training a neural network. Resnet50 is a 50-layer deep neural network using around 150+ layers to dig deep into the hole for the extensive training. The major breakthrough of this network is since the vanishing and exploding gradient descent problem occurs in a typical neural network due to concept of backpropagation for adjusting weights and bias, so in order to rectify this the authors have introduced this Residual network which uses Skip Connection solve the above problem.

![Resnet](https://iq.opengenus.org/content/images/2020/03/Screenshot-from-2020-03-20-15-49-54.png)

**Reference**

+ https://iq.opengenus.org/resnet50-architecture/
+ https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
+ https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd A:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![thanks](https://media1.tenor.com/images/11ae4fcfc41bb9e66a0176fcfc38e695/tenor.gif?itemid=8486985)
  
### _**Credits to my friend who gave detailed explanation of installation procedure.**_
+ https://github.com/PaVaNTrIpAtHi
+ https://www.linkedin.com/in/pavan-tripathi-3993641a1/
  
# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.

+ https://techieyantechnologies.com/2022/07/how-to-install-anaconda/

+ https://techieyantechnologies.com/2022/06/get-started-with-creating-new-environment-in-anaconda-configuring-jupyter-notebook-and-installing-libraries-using-requirements-txt-2/

1.	Install the prerequisites/software’s required to execute the code from reading the above blog which is provided in the link above.
2.	Press windows key and type in anaconda prompt a terminal opens up.
3.	Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5.	Install necessary libraries from requirements.txt file provided.
6.	Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7.	Run main_resnet50.ipynb and make sure to change the path where your executable files are located and also the model path in the code, please follow the link on how to install and set up anaconda environment to execute files.

# _**Data Description**_
The dataset used in this project was collected form the Kaggle data repository which is basically a hub for data scientists and all the enthusiasts who wants to do some exciting things with data. The dataset is basically divided into 2 classes, normal roads and roads with potholes. The data was already labelled and segregated into normal roads and pothole roads where each class consisting of about 350 normal road images and 329 pothole roads.

**Pothole**

![Potholes](https://media1.tenor.com/images/9a3acd7460d0310061bc5aef9f9ea319/tenor.gif?itemid=7885434)

**Normal**

![Normal](https://quicklol.com/wp-content/uploads/2017/03/ducks-crossing-road-animated-gif-image.gif)

 # _**Issues Faced.**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.
4. Refer to the Below link to get more details on installing python and anaconda and how to configure it.
https://techieyantechnologies.com/2022/06/get-started-with-creating-new-environment-in-anaconda-configuring-jupyter-notebook-and-installing-libraries-using-requirements-txt-2/
5. Loading an understanding the transfer learning concept might be tricky at the start. Please refer to the algorithm description to learn more about transfer learning.

# _**Note:**_
**All the required data hasn't been provided over here. Please feel free to contact me for any issues. You can also download the dataset from the given link below.**

+ https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset

**Credits to the owner who provided the dataset**

### _**Let’s Connect**_
https://www.linkedin.com/in/abhinay-lingala-5a3ab7205/

![Connect](https://media2.giphy.com/media/l1O6zvqu7O317887HF/source.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media.giphy.com/media/GK7grZYLG7cs0/giphy.gif)
  

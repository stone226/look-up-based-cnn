# look-up-based-cnn
This repository comes from https://github.com/ildoonet/tf-lcnn

I have trained the alexnet-lcnn with mnist(24,24) resized from (28,28),and upload the model.The file named myinference is similar the file named inference which in the https://github.com/ildoonet/tf-lcnn, mainly delete some codes about logging. 
There is a problem to be solved that if use sparse-conv,the predict result is always wrong,while using dense-conv ,the predict is right.
